import logging
from typing import List, Dict, Optional


import torch
import pickle
from ddsketch import DDSketch
from tdigest import TDigest
from flwr.client import NumPyClient
from flwr.common import Context, ConfigRecord, ArrayRecord, Array, Scalar
from lightning import seed_everything


from conformal_fairness.data.base_datamodule import BaseDataModule
from conformal_fairness.utils import get_label_scores
from conformal_fairness.constants import Stage, FairnessMetric
from conformal_fairness.cp_methods.scores import CPScore

from fed_dp import (
    advanced_composition_per_round,
    build_tree,
    hist_edges,
    hist_leaf_index,
    hist_num_levels,
    prefix_count_bounds,
    sample_nonneg_truncated_laplace,
    sample_truncated_laplace,
    truncated_laplace_params,
)
from fed_config import FedConfFairExptConfig
from base_fed_model.task import load_data, load_acs_data
from folktables_datamodule import FolktablesDataModule
from pokec_datamodule import PokecDataModule
from fed_constants import (
    CLIENT_ID,
    CommFormulations,
    QuantileMethod,
    SpecialParameters,
    FedCPFitConstants,
    FedCFConstants,
    FedConfFairStage,
    STATE_CONFIGS,
    STATE_PARAMS,
)
from fed_utils import FixedDAPSScore, get_filter_mask


class CPFedClient(NumPyClient):
    def __init__(
        self,
        context: Context,
        datamodule: BaseDataModule,
        calib_probs: torch.Tensor,
        partition_id: int,
        num_partitions: int,
        client_formulation: int = CommFormulations.MORE_PRIVATE.value,
        use_mle: bool = False,
        all_probs: Optional[torch.Tensor] = None,
        add_dp_noise: bool = False,
        dp_epsilon: float = 1.0,
        dp_delta: float = 1e-5,
        num_dp_rounds: int = 1,
        dp_release: str = "per_round",
        dp_hist_bins: int = 64,
        dp_hist_max_score: float = 1.0,
    ):
        self.datamodule: BaseDataModule = datamodule
        self.partition_id: int = partition_id
        self.num_partitions: int = num_partitions
        self.client_formulation = client_formulation
        self.calib_probs = calib_probs
        self.use_mle = use_mle
        self.all_probs = all_probs
        self.add_dp_noise = add_dp_noise
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.num_dp_rounds = num_dp_rounds
        self.dp_release = dp_release
        self.dp_hist_bins = dp_hist_bins
        self.dp_hist_max_score = dp_hist_max_score

        # number of calibration points
        self.n_k: int = int(self.datamodule.split_dict[Stage.CALIBRATION].shape[0])

        # Ensuring probs is only the size of calibration points
        assert (
            self.n_k == self.calib_probs.shape[0]
        ), f"The number of client calibration points {self.n_k} is different than the number of calibration probs {self.calib_probs.shape[0]}"

        # This dictonary allows the clients to be stateful (i.e., retain information from previous rounds)
        self.client_state = context.state
        if STATE_CONFIGS not in self.client_state.config_records:
            self.client_state.config_records[STATE_CONFIGS] = ConfigRecord()
        if STATE_PARAMS not in self.client_state.parameters_records:
            self.client_state.parameters_records[STATE_PARAMS] = ArrayRecord()

    # The following getters and setters make it easier to store/save the stateful values
    @property
    def calib_labels(self):
        return self.datamodule.y[self.datamodule.split_dict[Stage.CALIBRATION]]

    @property
    def calib_groups(self):
        return self.datamodule.sens[self.datamodule.split_dict[Stage.CALIBRATION]]

    @property
    def fairness_metric(self):
        if "f_metric" not in self.client_state.config_records[STATE_CONFIGS]:
            return None
        return self.client_state.config_records[STATE_CONFIGS]["f_metric"]

    @fairness_metric.setter
    def fairness_metric(self, value):
        self.client_state.config_records[STATE_CONFIGS]["f_metric"] = value

    @property
    def _dp_noisy_tree(self):
        if "dp_noisy_tree" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["dp_noisy_tree"].numpy()
        )

    @_dp_noisy_tree.setter
    def _dp_noisy_tree(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["dp_noisy_tree"] = Array(
            value.detach().cpu().numpy()
        )

    def _hist_dp_params(self):
        """(m, b, A) for the one-shot tree release (see fed_dp.py)."""
        n_filters = 2 if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value else 1
        m = 2 * n_filters * self.positive_labels.shape[0] * hist_num_levels(self.dp_hist_bins)
        b, A = truncated_laplace_params(1.0, self.dp_epsilon / m, self.dp_delta / m)
        return m, b, A

    def _release_noisy_tree(self):
        """Build and noise the score histograms once; later rounds reuse them.

        Shape: (n_filters, num_groups, num_classes, 2B). Only positive labels
        present at the first coverage-gap round are filled; labels are indexed by
        class id so later label convergence does not change the layout.
        """
        _, b, A = self._hist_dp_params()
        edges = hist_edges(self.dp_hist_bins, self.dp_hist_max_score)
        B = self.dp_hist_bins
        n_filters = 2 if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value else 1
        num_groups = self.datamodule.num_sensitive_groups
        num_classes = self._cached_scores.shape[1]
        leaves = torch.zeros((n_filters, num_groups, num_classes, B), dtype=torch.float64)
        for pos in self.positive_labels.tolist():
            leaf = hist_leaf_index(self._cached_scores[:, pos], edges)
            for g_id in range(num_groups):
                masks = self._get_filter_mask(pos, g_id)
                if n_filters == 1:
                    masks = (masks,)
                for f, mask in enumerate(masks):
                    idx = leaf[mask]
                    idx = idx[idx < B]  # scores above max_score are never counted
                    leaves[f, g_id, pos] = torch.bincount(idx, minlength=B).to(torch.float64)
        tree = build_tree(leaves)
        tree[..., 1:] += sample_truncated_laplace(tree[..., 1:].shape, b, A)
        self._dp_noisy_tree = tree
        # A depends on |Y+| at release time, which shrinks as labels converge
        self.client_state.config_records[STATE_CONFIGS]["dp_hist_A"] = float(A)
        return tree

    def _hist_alpha_bounds(self, f: int, g_id: int, pos: int, lmbda: float, n_total: float):
        tree = self._dp_noisy_tree
        if tree is None:
            tree = self._release_noisy_tree()
        A = float(self.client_state.config_records[STATE_CONFIGS]["dp_hist_A"])
        edges = hist_edges(self.dp_hist_bins, self.dp_hist_max_score)
        return prefix_count_bounds(tree[f, g_id, int(pos)], A, edges, float(lmbda), n_total)

    @property
    def _cached_scores(self):
        if "cached_scores" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["cached_scores"].numpy()
        )

    @_cached_scores.setter
    def _cached_scores(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["cached_scores"] = Array(
            value.detach().cpu().numpy()
        )

    @property
    def positive_labels(self):
        if "pos_labels" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["pos_labels"].numpy()
        )

    @positive_labels.setter
    def positive_labels(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["pos_labels"] = Array(
            value.detach().cpu().numpy()
        )

    @property
    def prior(self):
        if "prior" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["prior"].numpy()
        )

    @prior.setter
    def prior(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["prior"] = Array(
            value.detach().cpu().numpy()
        )

    @property
    def U(self):
        if "U" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["U"].numpy()
        )

    @U.setter
    def U(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["U"] = Array(
            value.detach().cpu().numpy()
        )

    @property
    def L(self):
        if "L" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["L"].numpy()
        )

    @L.setter
    def L(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["L"] = Array(
            value.detach().cpu().numpy()
        )

    @property
    def prior_2(self):
        if "prior_2" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["prior_2"].numpy()
        )

    @prior_2.setter
    def prior_2(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["prior_2"] = Array(
            value.detach().cpu().numpy()
        )

    @property
    def U_2(self):
        if "U_2" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["U_2"].numpy()
        )

    @U_2.setter
    def U_2(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["U_2"] = Array(
            value.detach().cpu().numpy()
        )

    @property
    def L_2(self):
        if "L_2" not in self.client_state.parameters_records[STATE_PARAMS]:
            return None
        return torch.from_numpy(
            self.client_state.parameters_records[STATE_PARAMS]["L_2"].numpy()
        )

    @L_2.setter
    def L_2(self, value):
        self.client_state.parameters_records[STATE_PARAMS]["L_2"] = Array(
            value.detach().cpu().numpy()
        )

    def fit(self, parameters, config):
        """Train the model with data of this client."""

        assert FedConfFairStage.key in config.keys(), "Need to know the fed_cf stage"

        fed_cf_stage = config[FedConfFairStage.key]
        assert (
            fed_cf_stage in FedConfFairStage.values
        ), f"Stage, {fed_cf_stage}, is not a valid stage"
        # The stage value (sent by the server config) indicates which stage of FedCP/FedCF we are in
        # This allows us to differentiate the training methods
        if fed_cf_stage == FedConfFairStage.FED_CP:
            p, n, m = self.fed_cp_fit(parameters, config)

        elif fed_cf_stage == FedConfFairStage.POP_STAT:
            # Assume fed_cp has already been run
            p, n, m = self.fed_cf_pop_stat(parameters, config)

        elif fed_cf_stage == FedConfFairStage.CF_ITER:
            if SpecialParameters.key in config.keys():
                # check what addition param is there
                self.update_from_special_parameters(parameters, config)

            p, n, m = self.fed_cf_local_cg(parameters, config)

        m.update(
            {CLIENT_ID: self.partition_id}
        )  # Adding client ID to metrics for logging later
        return p, n, m

    def fed_cp_fit(self, parameters, config: Dict[str, Scalar]):
        """
        Computes the Scores and a the client sketch
        config - it should contain a 'quantile_method' in its key
        parameters - dummy variable to match function signature
        """
        assert (
            QuantileMethod.key in config.keys()
        ), "Expected quantile_method in configs"
        assert (
            FedCPFitConstants.SCORE_MODULE in config.keys()
        ), "Expected score_module in configs"

        score_module: CPScore = pickle.loads(config[FedCPFitConstants.SCORE_MODULE])
        if isinstance(score_module, FixedDAPSScore):
            self._cached_scores = score_module.compute(
                self.all_probs, datamodule=self.datamodule
            )[self.datamodule.split_dict[Stage.CALIBRATION]]
        else:
            self._cached_scores = score_module.pipe_compute(self.calib_probs)

        # Mask is all since we only have the probs for the calibration datamodule
        label_scores = get_label_scores(
            labels=self.calib_labels,
            scores=self._cached_scores,
            mask=torch.ones(self.n_k, dtype=torch.bool),
            dataset=self.datamodule.name,
        )

        quantile_method = config[QuantileMethod.key]

        """
        This code is based upon the github repo provided by FedCP.
        Modified from https://github.com/clu5/federated-conformal/blob/main/src/conformal.py
        """
        # Compute the specific skethc - pickling the sketch since we
        # can't directly send the sketch over
        if quantile_method == QuantileMethod.TDIGEST:
            client_digest = TDigest()
            client_digest.batch_update(label_scores.numpy())
            sketch = pickle.dumps(client_digest)

        elif quantile_method == QuantileMethod.DDSKETCH:
            client_sketch = DDSketch()
            for score in label_scores.tolist():
                client_sketch.add(score)
            sketch = pickle.dumps(client_sketch)

        elif quantile_method == QuantileMethod.MEAN:
            quantile = score_module.compute_quantile(label_scores)
            sketch = float(quantile)

        else:
            raise ValueError(f"{quantile_method} not supported")

        # Sending the sketch over as a metric (which is a dict)
        # since it is not a parameters
        metrics = {FedCPFitConstants.QUANT_SKETCH: sketch}
        num_examples = self.n_k

        return [], num_examples, metrics

    def fed_cf_pop_stat(self, parameters, config: Dict[str, Scalar]):
        """
        Computes the population statistics w.r.t the fairness metric
        It is used as a prior in our computations.
        Here we only compute the lower bound. The server has enough information to compute the upper bound from the lower
        config - needs to have the fairness metric
        parameters - should only have one tensor which is the positive labels (assuming 0 indexing)
        """
        assert (
            FedCFConstants.F_M in config.keys()
        ), "The fairness metric needs to be provided to the client to compute population stats."

        self.fairness_metric = config[FedCFConstants.F_M]
        self.positive_labels = torch.from_numpy(parameters[0])

        # Empty matrices to store pop stats computations
        prior = torch.zeros(
            (self.datamodule.num_sensitive_groups, self.positive_labels.shape[0])
        )
        if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
            prior_pe = torch.zeros(
                (self.datamodule.num_sensitive_groups, self.positive_labels.shape[0])
            )
        for i, pos in enumerate(self.positive_labels):
            for g_id in range(self.datamodule.num_sensitive_groups):
                f_m = self._get_filter_mask(pos, g_id)
                if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                    # for equalized odds prior is equal oppportunity
                    # and prior_pe for predictive equality

                    if self.use_mle:
                        prior[g_id, i] = sum(f_m[0]) / (self.n_k)
                        prior_pe[g_id, i] = sum(f_m[1]) / (self.n_k)
                    else:
                        prior[g_id, i] = sum(f_m[0]) / (self.n_k + 1)
                        prior_pe[g_id, i] = sum(f_m[1]) / (self.n_k + 1)
                else:
                    if self.use_mle:
                        prior[g_id, i] = sum(f_m) / (self.n_k)
                    else:
                        prior[g_id, i] = sum(f_m) / (self.n_k + 1)

        nd_array = [prior.cpu().detach().numpy()]
        if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
            nd_array = nd_array + [prior_pe.cpu().detach().numpy()]

        # The tensor containing the lower_bound must be coverted to a parameter.
        # param = ndarrays_to_parameters(nd_array)
        num_examples = self.n_k
        metrics = {FedCFConstants.CLIENT_FORM: self.client_formulation}

        return nd_array, num_examples, metrics

    def fed_cf_local_cg(self, parameters, config):
        """
        The parameters includes the new set of lambdas
        """
        curr_lmbda = torch.from_numpy(parameters[0])

        # Handle group-aware lambdas
        use_group_info = config.get("use_group_information", False)
        if use_group_info:
            # Reshape flattened lambda back to (num_groups, num_active_pos_labels)
            num_groups = self.datamodule.num_sensitive_groups
            num_active_labels = curr_lmbda.numel() // num_groups
            curr_lmbda = curr_lmbda.reshape(num_groups, num_active_labels)

        if self.partition_id == 0:
            logging.info(
                f"[Client {self.partition_id}/{self.num_partitions}] Current lambda: {curr_lmbda}"
            )

        # Handle group-aware dynamic label sizing due to label convergence
        # In group-aware mode, the server may send fewer labels than self.positive_labels
        # if some labels have converged. Infer the number of active labels from the received lambda shape.
        if use_group_info:
            num_active_labels = curr_lmbda.shape[1]
            active_pos_label_indices = list(range(num_active_labels))
            score_below = self._cached_scores[
                :, self.positive_labels[active_pos_label_indices]
            ]
        else:
            # Non-group-aware: curr_lmbda already has the correct active label count
            score_below = self._cached_scores[:, self.positive_labels]

        # For group-aware: compute score_below per group
        if use_group_info:
            # score_below_dict will map group_id -> (n_k, num_active_labels) boolean tensor
            score_below_dict = {}
            for g_id in range(self.datamodule.num_sensitive_groups):
                score_below_dict[g_id] = score_below <= curr_lmbda[g_id].unsqueeze(0)
        else:
            score_below = score_below <= curr_lmbda

        final_nd_arrays = None

        lower = torch.zeros(
            (
                self.datamodule.num_sensitive_groups,
                (
                    self.positive_labels.shape[0]
                    if not use_group_info
                    else num_active_labels
                ),
            )
        )
        upper = torch.zeros(
            (
                self.datamodule.num_sensitive_groups,
                (
                    self.positive_labels.shape[0]
                    if not use_group_info
                    else num_active_labels
                ),
            )
        )
        if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
            lower_pe = torch.zeros(
                (
                    self.datamodule.num_sensitive_groups,
                    (
                        self.positive_labels.shape[0]
                        if not use_group_info
                        else num_active_labels
                    ),
                )
            )
            upper_pe = torch.zeros(
                (
                    self.datamodule.num_sensitive_groups,
                    (
                        self.positive_labels.shape[0]
                        if not use_group_info
                        else num_active_labels
                    ),
                )
            )

        # Loop to compute coverages for each group - label pair
        # TODO: make more efficient for demographic parity
        # In group-aware mode with label convergence, iterate only over active labels
        if use_group_info:
            labels_to_iterate = [
                (i, self.positive_labels[active_pos_label_indices[i]])
                for i in range(num_active_labels)
            ]
        else:
            labels_to_iterate = list(enumerate(self.positive_labels))

        use_hist_dp = self.add_dp_noise and self.dp_release == "histogram"

        def alpha_bounds(f_idx, mask, g_id, i, pos, score_below_curr):
            """(lower, upper) on #{calibration points in mask with score <= lambda}.
            Exact without DP; from the one-shot noisy tree with histogram DP."""
            if not use_hist_dp:
                a = sum(score_below_curr[mask, i])
                return a, a
            lmbda = curr_lmbda[g_id, i] if use_group_info else curr_lmbda[i]
            return self._hist_alpha_bounds(f_idx, g_id, pos, lmbda, float(sum(mask)))

        for i, pos in labels_to_iterate:
            for g_id in range(self.datamodule.num_sensitive_groups):
                # The validity of these calculations can be seen in the overleaf
                if use_group_info:
                    score_below_curr = score_below_dict[g_id]
                else:
                    score_below_curr = score_below

                if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                    f_eo, f_pe = self._get_filter_mask(pos, g_id)  # eo and pe
                    alpha_feo_lo, alpha_feo = alpha_bounds(0, f_eo, g_id, i, pos, score_below_curr)
                    alpha_fpe_lo, alpha_fpe = alpha_bounds(1, f_pe, g_id, i, pos, score_below_curr)

                    if self.use_mle:
                        # In this case upper and lower are the same since a point estimate is an interval with the same endpoints
                        upper[g_id, i] = alpha_feo / self.n_k
                        upper_pe[g_id, i] = alpha_fpe / self.n_k

                        lower[g_id, i] = alpha_feo_lo / self.n_k
                        lower_pe[g_id, i] = alpha_fpe_lo / self.n_k
                    else:
                        n_feo = sum(f_eo)
                        upper[g_id, i] = (alpha_feo + 1) / (self.n_k + 1)
                        lower[g_id, i] = (alpha_feo_lo * n_feo) / (
                            (n_feo + 1) * (self.n_k + 1)
                        )

                        n_fpe = sum(f_pe)
                        upper_pe[g_id, i] = (alpha_fpe + 1) / (self.n_k + 1)
                        lower_pe[g_id, i] = (alpha_fpe_lo * n_fpe) / (
                            (n_fpe + 1) * (self.n_k + 1)
                        )
                else:
                    f_m = self._get_filter_mask(pos, g_id)
                    alpha_fm_lo, alpha_fm = alpha_bounds(0, f_m, g_id, i, pos, score_below_curr)
                    if self.use_mle:
                        upper[g_id, i] = alpha_fm / self.n_k
                        lower[g_id, i] = alpha_fm_lo / self.n_k
                    else:
                        n_fm = sum(f_m)
                        upper[g_id, i] = (alpha_fm + 1) / (self.n_k + 1)
                        lower[g_id, i] = (alpha_fm_lo * n_fm) / (
                            (n_fm + 1) * (self.n_k + 1)
                        )

        if self.add_dp_noise and self.dp_release == "per_round":
            # Non-negative truncated Laplace noise: inflate upper, deflate lower,
            # so the server only ever overestimates the coverage gap (fed_dp.py).
            n_arrays = 4 if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value else 2
            # One replaced point changes at most 2 groups x |Y+| cells per array.
            m = 2 * self.positive_labels.shape[0] * n_arrays
            eps_0, delta_0 = advanced_composition_per_round(
                self.dp_epsilon, self.dp_delta, self.num_dp_rounds
            )
            b, A = truncated_laplace_params(1.0 / self.n_k, eps_0 / m, delta_0 / m)
            noise = lambda shape: sample_nonneg_truncated_laplace(shape, b, A)
            upper = upper + noise(upper.shape)
            lower = lower - noise(lower.shape)
            if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                upper_pe = upper_pe + noise(upper_pe.shape)
                lower_pe = lower_pe - noise(lower_pe.shape)

        # Based on the formulation return the pos-label coverage for each group label pair
        # Or send the pairwise differences
        if self.client_formulation == CommFormulations.MORE_PRIVATE.value:
            lower = lower / self.U
            upper = upper / self.L
            pairwise_cg = [
                (upper[:, None, :] - lower[None, :, :]).cpu().detach().numpy()
            ]  # Creates a tensor of size (num_groups, num_groups, num_pos_labels)

            if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                lower_pe = lower_pe / self.U_2
                upper_pe = upper_pe / self.L_2
                pairwise_cg = pairwise_cg + [
                    (upper_pe[:, None, :] - lower_pe[None, :, :]).cpu().detach().numpy()
                ]

            final_nd_arrays = pairwise_cg

        elif self.client_formulation == CommFormulations.LOW_OVERHEAD.value:
            nd_list = [lower.cpu().detach().numpy(), upper.cpu().detach().numpy()]
            if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                nd_list = nd_list + [
                    lower_pe.cpu().detach().numpy(),
                    upper_pe.cpu().detach().numpy(),
                ]
            final_nd_arrays = nd_list

        metrics = dict()
        num_examples = self.n_k

        return final_nd_arrays, num_examples, metrics

    def _get_filter_mask(self, pos_label, group_id):
        labels = self.calib_labels
        groups = self.calib_groups

        return get_filter_mask(
            self.fairness_metric, labels, groups, pos_label, group_id
        )

    def update_from_special_parameters(self, parameters, config) -> None:
        assert (
            SpecialParameters.key in config.keys()
        ), "Need to have 'special_params' to use this function."

        special_param = config[SpecialParameters.key]
        assert (
            special_param in SpecialParameters.values
        ), f"{special_param} parameters is not listed as a Special Parameter."
        """Update U and L or positive label values"""
        if special_param == SpecialParameters.UPDATE_U_L:
            # getting the U and L values if needed
            self.L = torch.from_numpy(parameters[-1])
            self.U = torch.from_numpy(parameters[-2])
            if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                self.L_2 = torch.from_numpy(parameters[-3])
                self.U_2 = torch.from_numpy(parameters[-4])

        elif special_param == SpecialParameters.UPDATE_POS:
            # updating the active positive labels here
            new_pos_labels = torch.tensor(parameters[-1])
            # values that are true are still pos labels
            # assert (False), f"The updated pos labels are f{new_pos_labels}"
            mask = torch.isin(self.positive_labels, new_pos_labels)
            self.positive_labels = new_pos_labels

            # reduce the values of L and U that are needed
            if self.client_formulation == CommFormulations.MORE_PRIVATE.value:
                self.L = self.L[:, mask]
                self.U = self.U[:, mask]

                if self.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                    self.L_2 = self.L_2[:, mask]
                    self.U_2 = self.U_2[:, mask]

        else:
            raise NotImplementedError

    def evaluate(self, parameters, config):
        """Evaluate the model on the data this client has."""
        """ Using eval to retreive the values of U and L"""

        return 0, 0, dict()
        # return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def fair_client_fn(
    context: Context,
    args: FedConfFairExptConfig,
    datamodule: BaseDataModule,
    probs: torch.Tensor,
    client_formulations: List[int],
    client_mapping: Optional[torch.Tensor] = None,
    global_masks=None,
    global_client_mapping=None,
):
    """Construct a Client that will be run in a ClientApp.
    Assume Federated DataModule Is provided by the client
    """

    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    seed_everything(args.seed + partition_id)

    if isinstance(datamodule, FolktablesDataModule):
        assert (
            client_mapping is not None
        ), "Expected client_mapping for FolktablesDataModule"

        probs_dict = {}
        for part_id in client_mapping.unique():
            id_index = client_mapping == part_id
            probs_dict[part_id.item()] = probs[id_index]

        fed_datamodule, part_probs = load_acs_data(
            datamodule,
            partition_type=args.folktables_partition_type,
            partition_id=partition_id,
            global_masks=global_masks,
            global_client_mapping=global_client_mapping,
            probs=probs_dict,
        )
    elif isinstance(datamodule, PokecDataModule):
        fed_datamodule, part_probs = datamodule.load_partition(partition_id, probs)
    else:
        fed_datamodule, part_probs = load_data(
            datamodule,
            num_partitions=num_partitions,
            partition_id=partition_id,
            probs=probs,
        )

    cl_form = client_formulations[partition_id]

    # Parse out the calibration probabilities from all the probs
    calib_part_probs = part_probs[fed_datamodule.split_dict[Stage.CALIBRATION]]

    return CPFedClient(
        context=context,
        datamodule=fed_datamodule,
        calib_probs=calib_part_probs,
        partition_id=partition_id,
        num_partitions=num_partitions,
        client_formulation=cl_form,
        use_mle=args.cf_opt.use_mle,
        all_probs=part_probs if args.conformal_method.lower() == "daps" else None,
        add_dp_noise=args.add_dp_noise,
        dp_epsilon=args.dp_epsilon,
        dp_delta=args.dp_delta,
        num_dp_rounds=args.cf_opt.num_opt_rounds,
        dp_release=args.dp_release,
        dp_hist_bins=args.dp_hist_bins,
        dp_hist_max_score=2.0 if args.conformal_method.lower() == "raps" else 1.0,
    ).to_client()
