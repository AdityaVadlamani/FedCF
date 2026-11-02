import math
from typing import Dict, Union
from os.path import join
import torch
from copy import deepcopy
from conformal_fairness.data import BaseDataModule, BaseDataset
from conformal_fairness.constants import (
    ACS_INCOME,
    ACS_EDUC,
    Stage,
    FairnessMetric,
    ConformalMethod,
)
from conformal_fairness.config import (
    PrimitiveScoreConfig,
)
from conformal_fairness.conformal_predictors import APSScore, TPSScore
from conformal_fairness.cp_methods.transformations import (
    DiffusionTransformation,
    RegularizationTransformation,
)
from fed_config import (
    FedConfFairExptConfig,
    FixedDiffusionConfig,
    FixedRegularizedConfig,
)
from pokec_datamodule import PokecDataModule


def _group_label_pair_counts(
    groups: torch.Tensor, labels: torch.Tensor
) -> Dict[tuple, int]:
    counts = {}
    for group_id in groups.unique().tolist():
        for label in labels.unique().tolist():
            n = int(((groups == group_id) & (labels == label)).sum().item())
            if n > 0:
                counts[(group_id, label)] = n
    return counts


def _log_pair_counts(title: str, counts: Dict[tuple, int]):
    print(f"[distribution shift] {title}", flush=True)
    for (group_id, label), n in sorted(counts.items()):
        print(
            f"[distribution shift]   (group={group_id}, label={label}): {n}",
            flush=True,
        )


def _select_affected_client_mask(
    datamodule: BaseDataModule,
    test_idx: torch.Tensor,
    conf: FedConfFairExptConfig,
    gen: torch.Generator,
) -> torch.Tensor:
    """Bool mask, aligned with test_idx, marking test points whose client was
    randomly selected to be hit by the shift. All-True when
    distribution_shift_client_fraction is 1.0 (i.e. every client is affected,
    matching the original global-shift behavior).
    """
    if conf.distribution_shift_client_fraction >= 1.0:
        return torch.ones(test_idx.shape[0], dtype=torch.bool)

    assert hasattr(datamodule, "client_mapping"), (
        "distribution_shift_client_fraction < 1.0 requires a datamodule that "
        "exposes client_mapping (currently only FolktablesDataModule, i.e. ACS "
        "Income / ACS Education)"
    )

    client_ids = datamodule.client_mapping[test_idx]
    unique_clients = client_ids.unique()
    num_affected = max(
        1,
        math.ceil(conf.distribution_shift_client_fraction * unique_clients.shape[0]),
    )
    shuffled_clients = unique_clients[
        torch.randperm(unique_clients.shape[0], generator=gen)
    ]
    affected_clients = shuffled_clients[:num_affected]
    mask = torch.isin(client_ids, affected_clients)

    print(
        f"[distribution shift] hitting {num_affected}/{unique_clients.shape[0]} "
        f"clients: {sorted(affected_clients.tolist())}",
        flush=True,
    )
    return mask


def apply_distribution_shift(
    datamodule: BaseDataModule, conf: FedConfFairExptConfig, probs: torch.Tensor = None
):
    """Mutates `datamodule`'s test split (and, for the gaussian_noise shift, `probs`)
    in place to mimic a distribution shift.

    Only affects `datamodule.split_dict[Stage.TEST]` / `datamodule.sens` / `probs`,
    all of which are read fresh by `get_test_dm_probs` (and
    `PokecDataModule.load_test_dm_probs`) right after this runs, so no other caller
    needs to change. `probs` is mutated in place (not returned) since it's a plain
    tensor passed by reference from the caller.
    """
    if not (
        conf.distribution_shift_subsample
        or conf.distribution_shift_shuffle_group
        or conf.distribution_shift_swap_extreme_groups
        or conf.distribution_shift_gaussian_noise
    ):
        return

    test_idx = datamodule.split_dict[Stage.TEST]
    groups = datamodule.sens[test_idx]
    labels = datamodule.y[test_idx]

    gen = torch.Generator().manual_seed(conf.conformal_seed or conf.seed)

    before_counts = _group_label_pair_counts(groups, labels)
    _log_pair_counts("(group, label) counts before shift", before_counts)

    affected_mask = _select_affected_client_mask(datamodule, test_idx, conf, gen)

    if conf.distribution_shift_shuffle_group:
        new_groups = groups.clone()
        affected_positions = affected_mask.nonzero(as_tuple=True)[0]
        sub_groups = groups[affected_positions]
        perm = torch.randperm(sub_groups.shape[0], generator=gen)
        new_groups[affected_positions] = sub_groups[perm]
        datamodule.sens[test_idx] = new_groups

        after_counts = _group_label_pair_counts(datamodule.sens[test_idx], labels)
        _log_pair_counts(
            "(group, label) counts after shuffling group labels", after_counts
        )
        return

    if conf.distribution_shift_swap_extreme_groups:
        group_ids, group_counts = groups.unique(return_counts=True)
        most_common_group = group_ids[group_counts.argmax()].item()
        least_common_group = group_ids[group_counts.argmin()].item()

        new_groups = groups.clone()
        swap_most = affected_mask & (groups == most_common_group)
        swap_least = affected_mask & (groups == least_common_group)
        new_groups[swap_most] = least_common_group
        new_groups[swap_least] = most_common_group
        datamodule.sens[test_idx] = new_groups

        after_counts = _group_label_pair_counts(datamodule.sens[test_idx], labels)
        _log_pair_counts(
            f"(group, label) counts after swapping groups "
            f"{most_common_group} <-> {least_common_group}",
            after_counts,
        )
        return

    if conf.distribution_shift_gaussian_noise:
        assert probs is not None, "gaussian_noise shift requires probs to be passed"

        affected_positions = affected_mask.nonzero(as_tuple=True)[0]
        affected_test_idx = test_idx[affected_positions]

        std = conf.distribution_shift_gaussian_noise_std
        noise = (
            torch.randn(affected_test_idx.shape[0], probs.shape[1], generator=gen) * std
        )
        noised = (probs[affected_test_idx] + noise).clamp(min=0)
        noised = noised / noised.sum(dim=1, keepdim=True).clamp(min=1e-12)
        probs[affected_test_idx] = noised

        print(
            f"[distribution shift] added N(0, {std}^2) noise to probs for "
            f"{affected_test_idx.shape[0]}/{test_idx.shape[0]} test points, "
            "then clipped to non-negative and renormalized",
            flush=True,
        )
        return

    min_points_per_pair = math.ceil(2 * conf.num_clients / conf.closeness_measure)

    keep_mask = torch.ones(test_idx.shape[0], dtype=torch.bool)
    for (group_id, label), n in before_counts.items():
        if n <= min_points_per_pair:
            continue
        pair_idx = ((groups == group_id) & (labels == label)).nonzero(as_tuple=True)[0]
        pair_idx = pair_idx[affected_mask[pair_idx]]
        if pair_idx.shape[0] == 0:
            continue
        # drop somewhere between 1 and (n - min) points, not all the way down to min,
        # capped by how many of those points belong to an affected client
        num_to_drop = int(
            torch.randint(1, n - min_points_per_pair + 1, (1,), generator=gen).item()
        )
        num_to_drop = min(num_to_drop, pair_idx.shape[0])
        drop_idx = pair_idx[torch.randperm(pair_idx.shape[0], generator=gen)][
            :num_to_drop
        ]
        keep_mask[drop_idx] = False

    kept_test_idx = test_idx[keep_mask]
    datamodule.split_dict[Stage.TEST] = kept_test_idx

    after_counts = _group_label_pair_counts(
        datamodule.sens[kept_test_idx], datamodule.y[kept_test_idx]
    )
    _log_pair_counts(
        f"(group, label) counts after subsampling (min {min_points_per_pair} per pair)",
        after_counts,
    )


def get_test_dm_probs(datamodule: BaseDataModule, probs: torch.Tensor):
    if isinstance(datamodule, PokecDataModule):
        return datamodule.load_test_dm_probs(probs)

    ds: BaseDataset = datamodule._base_dataset
    ds_test = deepcopy(ds)

    if isinstance(ds_test.X, list):
        ds_test.X = [datamodule.X[i] for i in datamodule.split_dict[Stage.TEST]]
    else:
        ds_test.X = datamodule.X[datamodule.split_dict[Stage.TEST]]
    ds_test.y = datamodule.y[datamodule.split_dict[Stage.TEST]]
    ds_test.sens = datamodule.sens[datamodule.split_dict[Stage.TEST]]
    ds_test.split_config.train = 0
    ds_test.split_config.valid = 0
    ds_test.split_config.calib = 0

    dm_test = deepcopy(datamodule)
    dm_test._base_dataset = ds_test
    dm_test._init_with_dataset(ds_test)
    probs = probs[datamodule.split_dict[Stage.TEST]]

    assert dm_test.num_points == probs.shape[0]
    return dm_test, probs


def get_filter_mask(
    fairness_metric, labels: torch.Tensor, groups: torch.Tensor, pos_label, group_id
):
    assert (
        labels.shape[0] == groups.shape[0]
    ), f"Got {labels.shape[0]} labels, but {groups.shape[0]} groups"

    match fairness_metric:
        case FairnessMetric.EQUAL_OPPORTUNITY.value:
            label_satisfied = labels == pos_label
            group_satisfied = groups == group_id
            return (label_satisfied & group_satisfied).reshape(-1)

        case FairnessMetric.PREDICTIVE_EQUALITY.value:
            label_not_satisfied = labels != pos_label
            group_satisfied = groups == group_id
            return (label_not_satisfied & group_satisfied).reshape(-1)

        case FairnessMetric.EQUALIZED_ODDS.value:
            label_satisfied = labels == pos_label
            label_not_satisfied = labels != pos_label
            group_satisfied = groups == group_id

            return (
                (label_satisfied & group_satisfied).reshape(-1),
                (label_not_satisfied & group_satisfied).reshape(-1),
            )
        case (
            FairnessMetric.DEMOGRAPHIC_PARITY.value
            | FairnessMetric.DISPARATE_IMPACT.value
            | FairnessMetric.OVERALL_ACC_EQUALITY.value
        ):
            return (groups == group_id).reshape(-1)
        case _:
            raise NotImplementedError(
                f"Filtering function not implemented for {fairness_metric}"
            )


def get_score_module(conformal_method, split_conf_input, alpha=None):
    if conformal_method in [ConformalMethod.TPS, ConformalMethod.APS]:
        assert isinstance(split_conf_input, PrimitiveScoreConfig)
    elif conformal_method == ConformalMethod.DAPS:
        assert isinstance(split_conf_input, FixedDiffusionConfig)
    elif conformal_method == ConformalMethod.RAPS:
        assert isinstance(split_conf_input, FixedRegularizedConfig)
    else:
        raise NotImplementedError

    if conformal_method == ConformalMethod.TPS:
        score_module = TPSScore(split_conf_input, alpha=alpha)
    elif conformal_method == ConformalMethod.APS:
        score_module = APSScore(split_conf_input, alpha=alpha)
    elif conformal_method == ConformalMethod.RAPS:
        score_module = FixedRAPSScore(split_conf_input, alpha=alpha)
    elif conformal_method == ConformalMethod.DAPS:
        score_module = FixedDAPSScore(split_conf_input, alpha=alpha)
    else:
        raise NotImplementedError

    return score_module


class FixedDAPSScore(APSScore):
    def __init__(self, config: PrimitiveScoreConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

    def compute(self, probs, **kwargs):
        scores = super().compute(probs, **kwargs)

        daps_params = {
            DiffusionTransformation.DAPS_LAMBDA: self.config.daps_lambda,
        } | kwargs

        transform_module = DiffusionTransformation(config=self.config)

        scores = transform_module.transform(
            scores,
            **daps_params,
        )

        return scores


class FixedRAPSScore(APSScore):
    def __init__(self, config: PrimitiveScoreConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config

    def compute(self, probs, **kwargs):
        scores = super().compute(probs, **kwargs)

        raps_params = {
            RegularizationTransformation.RAPS_K: self.config.raps_k,
            RegularizationTransformation.RAPS_LAMBDA: self.config.raps_lambda,
        } | kwargs

        transform_module = RegularizationTransformation(config=self.config)

        if self.config.raps_mod:
            assert (
                self.config.use_aps_epsilon == False
            ), "Should not use randomized APS with modified RAPS"

        scores = transform_module.transform(
            scores,
            probs,
            raps_modified=self.config.raps_mod,
            **raps_params,
        )

        return scores


def get_fair_configs_res_path(conf: FedConfFairExptConfig):
    conf_dict: Dict[str, Union[int, str, float]] = dict()

    conf_dict["seed"] = conf.conformal_seed or conf.seed
    conf_dict["c"] = conf.closeness_measure

    conf_dict["add_dp_noise"] = conf.add_dp_noise
    conf_dict["binary_sens"] = conf.dataset.binary_sens
    conf_dict["dp_epsilon"] = conf.dp_epsilon
    if conf.add_dp_noise:
        conf_dict["dp_mechanism"] = "truncated_laplace"
        conf_dict["dp_delta"] = conf.dp_delta
        conf_dict["dp_release"] = conf.dp_release
        if conf.dp_release == "histogram":
            conf_dict["dp_hist_bins"] = conf.dp_hist_bins
    conf_dict["distribution_shift_subsample"] = conf.distribution_shift_subsample
    conf_dict["distribution_shift_shuffle_group"] = conf.distribution_shift_shuffle_group
    conf_dict["distribution_shift_swap_extreme_groups"] = (
        conf.distribution_shift_swap_extreme_groups
    )
    conf_dict["distribution_shift_gaussian_noise"] = conf.distribution_shift_gaussian_noise
    conf_dict["distribution_shift_gaussian_noise_std"] = (
        conf.distribution_shift_gaussian_noise_std
    )
    conf_dict["distribution_shift_client_fraction"] = (
        conf.distribution_shift_client_fraction
    )

    out_file_name = ""
    if conf.conformal_method == ConformalMethod.APS.value:
        out_file_name = (
            "aps.csv" if conf.primitive_config.use_aps_epsilon else "aps_no_rand.csv"
        )
    elif conf.conformal_method == ConformalMethod.TPS.value:
        out_file_name = (
            "tps_classwise.csv"
            if conf.primitive_config.use_tps_classwise
            else "tps.csv"
        )
    elif conf.conformal_method == ConformalMethod.RAPS.value:
        conf_dict[RegularizationTransformation.RAPS_K] = (
            conf.regularization_config.raps_k
        )
        conf_dict[RegularizationTransformation.RAPS_LAMBDA] = (
            conf.regularization_config.raps_lambda
        )
        if conf.regularization_config.raps_mod:
            out_file_name = "raps_mod.csv"
        else:
            out_file_name = (
                "raps_rand_no_mod.csv"
                if conf.primitive_config.use_aps_epsilon
                else "raps_no_rand_no_mod.csv"
            )
    elif conf.conformal_method == ConformalMethod.DAPS.value:
        conf_dict[DiffusionTransformation.DAPS_LAMBDA] = (
            conf.diffusion_config.daps_lambda
        )
        out_file_name = "daps.csv"
    else:
        raise NotImplementedError

    ds_name = f"{conf.dataset.name}"

    if conf.dataset.sens_attrs:
        ds_name += "_".join([""] + conf.dataset.sens_attrs)

    out_dir = join(
        "analysis",
        conf.results_dir,
        ds_name,
        (
            f"{conf.num_clients}_clients"
            if ds_name not in (ACS_INCOME, ACS_EDUC)
            else conf.folktables_partition_type
        ),
        f"split_{conf.dataset_split_fractions.train}_{conf.dataset_split_fractions.valid}",
        conf.fairness_metric,
    )

    return conf_dict, out_dir, out_file_name
