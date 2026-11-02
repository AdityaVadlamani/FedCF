from dataclasses import dataclass, field
from conformal_fairness.config import (
    BaseExptConfig,
    ConfExptConfig,
    ConfFairExptConfig,
    DiffusionConfig,
    RegularizedConfig,
)
from typing import List, Union, Optional
from fed_constants import FOLKTABLES_ALL, FOLKTABLES_CONTINENTAL_ALL, CommFormulations


@dataclass
class FixedRegularizedConfig(RegularizedConfig):
    raps_k: int = field(default=3)
    raps_lambda: float = field(default=0.1)


@dataclass
class FixedDiffusionConfig(DiffusionConfig):
    daps_lambda: float = field(default=0.2)


@dataclass
class FedBaseExptConfig(BaseExptConfig):
    num_server_rounds: int = field(default=3)
    fraction_fit: float = field(default=0.1)
    num_clients: int = field(default=10)
    folktables_partition_type: str = field(default="")  # Changed from int to str

    def __post_init__(self):
        super().__post_init__()
        if "small" in self.folktables_partition_type:
            self.num_clients = 4
        elif "large" in self.folktables_partition_type:
            self.num_clients = 8
        elif self.folktables_partition_type == FOLKTABLES_CONTINENTAL_ALL:
            self.num_clients = 48
        elif self.folktables_partition_type == FOLKTABLES_ALL:
            self.num_clients = 51


@dataclass
class FedConfExptConfig(ConfExptConfig):
    num_clients: int = field(default=10)
    folktables_partition_type: str = field(default="")

    quantile_method: str = field(default="ddsketch")
    debug_mode: bool = field(default=False)

    # Forcing it to have fixed parameters
    regularization_config: Optional[FixedRegularizedConfig] = field(
        default_factory=FixedRegularizedConfig
    )

    # Forcing it to have fixed parameters
    diffusion_config: Optional[FixedDiffusionConfig] = field(
        default_factory=FixedDiffusionConfig
    )

    def __post_init__(self):
        super().__post_init__()
        if "small" in self.folktables_partition_type:
            self.num_clients = 4
        elif "large" in self.folktables_partition_type:
            self.num_clients = 8
        elif self.folktables_partition_type == FOLKTABLES_CONTINENTAL_ALL:
            self.num_clients = 48
        elif self.folktables_partition_type == FOLKTABLES_ALL:
            self.num_clients = 51

    #     if self.conformal_method == ConformalMethod.RAPS:
    #         assert(self.regularization_config isinstance(FixedRegularizedConfig))


@dataclass
class CFOptConfig:
    """optimization parameters for the cf framework"""

    num_opt_rounds: int = field(default=25)
    lr: float = field(default=0.01)
    momentum: float = field(default=0.5)
    use_mle: bool = field(default=False)
    use_descent_based_approach: bool = field(default=True)
    use_group_information: bool = field(default=False)


@dataclass
class FedConfFairExptConfig(FedConfExptConfig, ConfFairExptConfig):
    client_formulations: Union[List[int], int] = field(
        default=CommFormulations.LOW_OVERHEAD.value
    )
    cf_opt: CFOptConfig = field(default_factory=CFOptConfig)

    # Simulate a distribution shift on the test set (relative to train/calib).
    # subsample: for every (group, label) pair in the test set, randomly drop
    # points down to ceil(2 * num_clients / closeness_measure), leaving pairs
    # that start below that count untouched.
    distribution_shift_subsample: bool = field(default=False)
    # shuffle_group: randomly permute which sensitive group each test point
    # is assigned to, decoupling the group label from the rest of the point.
    distribution_shift_shuffle_group: bool = field(default=False)
    # swap_extreme_groups: relabel the most common sensitive group as the
    # least common one and vice versa, leaving all other groups untouched.
    distribution_shift_swap_extreme_groups: bool = field(default=False)
    # gaussian_noise: add Gaussian noise (std distribution_shift_gaussian_noise_std)
    # to each test point's probability vector, then clip to be non-negative and
    # renormalize. Probs (not raw features) are the only artifact that flows into
    # conformal scores in this pipeline -- they're loaded pre-computed from a frozen
    # model checkpoint and never recomputed from X -- so this is applied directly to
    # probs as a proxy for "the model saw a noisy version of this point's features".
    distribution_shift_gaussian_noise: bool = field(default=False)
    distribution_shift_gaussian_noise_std: float = field(default=0.1)

    # distribution_shift_client_fraction: restricts whichever shift above is
    # enabled to a random subset of clients (selected via client_mapping),
    # leaving the rest of the clients' test points untouched. 1.0 (default)
    # applies the shift to every client, matching the old global behavior.
    # Only supported for datamodules that expose a client_mapping (currently
    # FolktablesDataModule, i.e. ACS Income / ACS Education).
    distribution_shift_client_fraction: float = field(default=1.0)

    # add_dp_noise: make each client's coverage-gap releases (eps, delta)-DP over
    # the whole run. dp_epsilon / dp_delta are TOTAL budgets, split across the
    # num_opt_rounds rounds with advanced composition; each released coordinate
    # gets non-negative truncated Laplace noise (added to upper terms, subtracted
    # from lower terms). See fed_dp.py.
    add_dp_noise: bool = field(default=False)
    dp_epsilon: float = field(default=1.0)
    dp_delta: float = field(default=1e-5)
    # dp_release: "per_round" noises every coverage-gap round (composition over
    # num_opt_rounds); "histogram" releases one noisy prefix-sum tree of each
    # client's calibration scores (dp_hist_bins bins) and answers every round
    # from it, so no composition over rounds is needed.
    dp_release: str = field(default="per_round")
    dp_hist_bins: int = field(default=64)

    # results_dir: folder under analysis/ that result CSVs are written to.
    results_dir: str = field(default="fairness_trials")

    def __post_init__(self):
        super().__post_init__()
        if isinstance(self.client_formulations, int):
            self.client_formulations = [self.client_formulations] * self.num_clients
        else:
            assert len(self.client_formulations) == self.num_clients
        assert (
            sum(
                [
                    self.distribution_shift_subsample,
                    self.distribution_shift_shuffle_group,
                    self.distribution_shift_swap_extreme_groups,
                    self.distribution_shift_gaussian_noise,
                ]
            )
            <= 1
        ), (
            "Choose at most one of distribution_shift_subsample / "
            "distribution_shift_shuffle_group / distribution_shift_swap_extreme_groups / "
            "distribution_shift_gaussian_noise"
        )
        assert self.distribution_shift_gaussian_noise_std >= 0, (
            "distribution_shift_gaussian_noise_std must be non-negative"
        )
        assert 0 < self.distribution_shift_client_fraction <= 1.0, (
            "distribution_shift_client_fraction must be in (0, 1]"
        )
        if self.distribution_shift_client_fraction < 1.0:
            assert (
                self.distribution_shift_subsample
                or self.distribution_shift_shuffle_group
                or self.distribution_shift_swap_extreme_groups
                or self.distribution_shift_gaussian_noise
            ), (
                "distribution_shift_client_fraction < 1.0 has no effect unless one of "
                "the distribution_shift_* flags is also enabled"
            )
        if self.add_dp_noise:
            assert all(
                cf == CommFormulations.LOW_OVERHEAD.value
                for cf in self.client_formulations
            ), "add_dp_noise currently only supports client_formulations=0 (LOW_OVERHEAD) for all clients"
            assert self.dp_release in ("per_round", "histogram"), "dp_release must be per_round or histogram"
