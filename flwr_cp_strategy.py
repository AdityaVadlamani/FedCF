import logging
import os
import pickle
from copy import deepcopy
from math import isclose
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from ddsketch import DDSketch
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy
from tdigest import TDigest

from conformal_fairness.constants import ConformalMethod, FairnessMetric, Stage
from conformal_fairness.cp_methods.scores import CPScore
from conformal_fairness.cp_methods.transformations import PredSetTransformation
from conformal_fairness.data import BaseDataModule
from conformal_fairness.utils import get_label_scores, get_split_conf
from conformal_fairness.utils.conf_utils import calc_coverage, calc_efficiency
from fed_config import FedConfExptConfig, FedConfFairExptConfig
from fed_dp import snap_to_grid
from fed_constants import (
    CommFormulations,
    FedCFConstants,
    FedConfFairStage,
    FedCPFitConstants,
    QuantileMethod,
    SpecialParameters,
)
from fed_utils import (
    FixedDAPSScore,
    get_fair_configs_res_path,
    get_filter_mask,
    get_score_module,
)


def append_row_by_column_name(row: pd.DataFrame, path: str):
    """Append `row` to the CSV at `path`, matching columns by name.

    Plain `to_csv(mode="a")` writes by position, so if the code now logs
    different columns than the file's header, rows silently misalign. Here, if
    the row's columns fit the existing header we append in header order;
    otherwise we rewrite the file with the union of columns. A file lock keeps
    concurrent Slurm jobs from interleaving writes.
    """
    import fcntl

    with open(path + ".lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            row.to_csv(path)
            return
        header = list(pd.read_csv(path, nrows=0, index_col=0).columns)
        if set(row.columns) <= set(header):
            row.reindex(columns=header).to_csv(path, mode="a", header=False)
        else:
            existing = pd.read_csv(path, index_col=0)
            pd.concat([existing, row], ignore_index=True).to_csv(path)


class FedCPStrategy(Strategy):
    def __init__(
        self,
        config: FedConfExptConfig,
        test_dm: BaseDataModule,
        test_probs: torch.Tensor,
        all_probs: torch.Tensor = None,
    ):
        self.quantile_method: str = config.quantile_method
        self.alpha: float = config.alpha
        self.q_hat: float = -1  # unit value
        self.N: int = -1
        self.K: int = config.num_clients

        # Tracks, per client id, the number of bytes sent to the server in each
        # round that client participated in (used to report communication overhead).
        self.client_bytes_sent: Dict[str, List[int]] = {}

        self.test_dm: BaseDataModule = test_dm
        self.test_probs: torch.Tensor = test_probs

        alpha_quant = (
            self.alpha if self.quantile_method == QuantileMethod.MEAN else None
        )
        self.split_conf_input = get_split_conf(config)

        self.score_module: CPScore = get_score_module(
            conformal_method=config.conformal_method,
            split_conf_input=self.split_conf_input,
            alpha=alpha_quant,
        )

        if (test_dm is not None) and (self.test_probs is not None):
            assert test_probs.shape[0] == test_dm.split_dict[Stage.TEST].shape[0]
            if isinstance(self.score_module, FixedDAPSScore):
                self.test_scores = self.score_module.compute(
                    all_probs, datamodule=test_dm
                )[test_dm.split_dict[Stage.TEST]]
            else:
                self.test_scores = self.score_module.compute(
                    test_probs, datamodule=test_dm
                )
        else:
            self.test_scores: torch.Tensor = None

        assert (
            self.quantile_method in QuantileMethod.values
        ), f"The Quantile Method, '{self.quantile_method}', is not supported."

    def _track_bytes_sent(self, results: List[Tuple[ClientProxy, FitRes]]) -> None:
        """Record the number of bytes each client sent to the server this round."""
        for client_proxy, fit_res in results:
            num_bytes = sum(
                arr.nbytes for arr in parameters_to_ndarrays(fit_res.parameters)
            )
            num_bytes += len(pickle.dumps(fit_res.metrics))
            self.client_bytes_sent.setdefault(client_proxy.cid, []).append(num_bytes)

    @property
    def test_label_scores(self):
        if self.test_scores is None:
            return None

        return get_label_scores(
            self.test_y,
            self.test_scores,
            torch.ones_like(self.test_y, dtype=torch.bool),
            self.test_dm.name,
        )

    @property
    def test_y(self):
        return self.test_dm.y[self.test_dm.split_dict[Stage.TEST]]

    @property
    def test_sens(self):
        return self.test_dm.sens[self.test_dm.split_dict[Stage.TEST]]

    def initialize_parameters(
        self, client_manager: ClientManager
    ) -> Optional[Parameters]:
        """Initialize the (global) model parameters."""
        pass

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        clients = client_manager.all()
        # We are useing all of the clients for fed_cp
        config = {
            FedConfFairStage.key: FedConfFairStage.FED_CP,
            QuantileMethod.key: self.quantile_method,
            FedCPFitConstants.SCORE_MODULE: pickle.dumps(self.score_module),
        }

        return [(client, FitIns(parameters, config)) for client in clients.values()]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""

        """
        This code is based upon the github repo provided by FedCP.
        Modified from https://github.com/clu5/federated-conformal/blob/main/src/conformal.py
        """

        if self.quantile_method == QuantileMethod.TDIGEST:
            digest = TDigest()
        elif self.quantile_method == QuantileMethod.DDSKETCH:
            sketch = DDSketch()
        elif self.quantile_method == QuantileMethod.MEAN:
            mean_q = 0
        else:
            raise ValueError(f"{self.quantile_method} not supported")

        N_k = [0] * self.K

        # Based on the agreed upon sketch type we aggregate here (after the client calls)
        for idx, res in enumerate(results):
            N_k[idx] = res[1].num_examples
            if self.quantile_method == QuantileMethod.TDIGEST:
                client_digest: TDigest = pickle.loads(
                    res[1].metrics[FedCPFitConstants.QUANT_SKETCH]
                )
                digest = digest + client_digest
            elif self.quantile_method == QuantileMethod.DDSKETCH:
                client_sketch: DDSketch = pickle.loads(
                    res[1].metrics[FedCPFitConstants.QUANT_SKETCH]
                )
                sketch.merge(client_sketch)
            elif self.quantile_method == QuantileMethod.MEAN:
                mean_q += res[1].metrics[FedCPFitConstants.QUANT_SKETCH]

        self.N = sum(N_k)
        t = np.ceil((self.N + self.K) * (1 - self.alpha)) / self.N
        if self.quantile_method == QuantileMethod.TDIGEST:
            self.q_hat = digest.percentile(round(100 * t))
        elif self.quantile_method == QuantileMethod.DDSKETCH:
            self.q_hat = sketch.get_quantile_value(t)
        elif self.quantile_method == QuantileMethod.MEAN:
            self.q_hat = mean_q / self.K

        # unify the type to torch.tensor
        self.q_hat = torch.tensor(self.q_hat)
        assert self.q_hat is not None

        logging.info(f"FedCP: q_hat: {self.q_hat}")
        metrics_aggregated = {FedCPFitConstants.Q_HAT: self.q_hat}
        empty_parameters = Parameters(tensors=[], tensor_type="")
        return empty_parameters, metrics_aggregated

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        return (None, {})

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate the current model parameters."""

        if self.test_scores is None:
            return None

        prediction_sets = PredSetTransformation(threshold=self.q_hat).pipe_transform(
            self.test_scores
        )

        cov = calc_coverage(prediction_sets=prediction_sets, labels=self.test_y)
        eff = calc_efficiency(prediction_sets=prediction_sets)

        metrics = {"fed_cp_coverage": cov, "fed_cp_efficiency": eff}
        return cov, metrics


class FedConfFairStrategy(FedCPStrategy):
    def __init__(
        self,
        config: FedConfFairExptConfig,
        test_dm: BaseDataModule,
        test_probs: torch.Tensor,
        positive_labels: torch.Tensor,
        all_probs: torch.Tensor = None,
    ):
        """
        This Strategy is setup like a state machine.
        The first state computes the Federated Conformal Predictor (via inheritance)
        Then we move to the second state which computes the population stats.
        Lastly, the third stage is our optimization framework for conformal fairness.
        The last stage repeats for the number of specified rounds.
        config - necesssary parameters to run a FedConfFair framework
        num_classes - This information is only in the datamodule so
                        it is included here outside of the config.
        num_sens - The number of groups in the dataset.
        positive_labels - The positive labels/outcomes
        """

        super().__init__(
            config=config, test_dm=test_dm, test_probs=test_probs, all_probs=all_probs
        )
        self.config = config
        self.stage = FedConfFairStage.INIT_STAGE
        self.num_classes = self.test_dm.num_classes
        self.num_sens = self.test_dm.num_sensitive_groups
        self.positive_labels = positive_labels
        self.sensitive_groups = torch.unique(self.test_dm.sens)
        self.lmbda_upper_bound = (
            2 if config.conformal_method == ConformalMethod.RAPS else 1
        )

        # active_pos_label is always just the labels (not group-aware), shared in both modes
        self.active_pos_label = deepcopy(self.positive_labels)

        # Early stopping is 1D in both modes (indexed by label only)
        self.early_stop_label = torch.ones(self.num_classes, dtype=torch.bool)
        self.early_stop_label[positive_labels] = False

        if config.cf_opt.use_group_information:
            self.opt_lmbda = torch.ones((self.num_sens, self.num_classes))
            self.curr_lmbda = (
                torch.ones_like(positive_labels, dtype=torch.float)
                .unsqueeze(0)
                .repeat(self.num_sens, 1)
            )
            self.b_curr = torch.zeros_like(self.curr_lmbda, dtype=torch.float)
        else:
            self.opt_lmbda = torch.ones((self.num_classes,))
            self.curr_lmbda = torch.ones_like(positive_labels, dtype=torch.float)
            self.b_curr = torch.zeros_like(self.curr_lmbda, dtype=torch.float)

        # Used for debugging - stores coverage gap for each active label
        # Note: cg is aggregated across groups, so it's 1D
        self.active_cg = torch.zeros(len(self.positive_labels), dtype=torch.float)

        self.cf_round = 0

        # Init population stats for the fairness metric (values filled in POP_STAT stage)
        self.U = None
        self.L = None

        # Only init and use second set if we are using Equalized Odds or a metric with a second component
        self.U_2 = None
        self.L_2 = None

        # Client Formulations - which overhead paradigm are they using
        self.client_formulations: Dict[str, int] = dict()

        self.all_low_overhead_formulation = False

        self.num_rounds = (
            config.cf_opt.num_opt_rounds + 2
        )  # 1 for the fed-cf stage, and one for pop-stat

        self.update_pos_label_client = False

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        # Update the current stage - done here since this function will always be called
        # and it is the start of the training iteration
        self.__switch_state()
        """Configure the next round of training."""
        if self.stage == FedConfFairStage.FED_CP:
            return super().configure_fit(server_round, parameters, client_manager)
        elif self.stage == FedConfFairStage.POP_STAT:
            return self._pop_stat_configure_fit(
                server_round, parameters, client_manager
            )
        elif self.stage == FedConfFairStage.CF_ITER:
            # Check if all active labels have stopped (early_stop_label is now 1D in both modes)
            all_stopped = torch.all(self.early_stop_label[self.active_pos_label])

            if len(self.active_pos_label) == 0 or all_stopped:
                return []  # "Early stopping" by sampling no clients

            return self._satisfy_lambda_configure_fit(
                server_round, parameters, client_manager
            )
        else:
            raise NotImplementedError

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate training results."""
        if failures:
            # A round with missing clients gives a wrong coverage gap; fail
            # loudly instead of writing a plausible-looking result.
            raise RuntimeError(
                f"Round {server_round}: {len(failures)} client(s) failed, "
                f"e.g. {failures[0]!r}"
            )
        self._track_bytes_sent(results)

        if self.stage == FedConfFairStage.FED_CP:
            p, m = super().aggregate_fit(
                server_round, results, failures
            )  # updates self.q_hat
            self.curr_lmbda *= self.q_hat  # Initialize the fair_lambdas here

            # Opt lambda is lmbda_upper_bound for positive labels (and is the minimum q_hat for everything else)
            if self.config.cf_opt.use_group_information:
                # For group-aware: initialize opt_lmbda for each group (2D tensor)
                self.opt_lmbda *= self.q_hat
                self.opt_lmbda[:, self.positive_labels] = self.lmbda_upper_bound
            else:
                # For non-group: standard initialization
                self.opt_lmbda *= self.q_hat
                self.opt_lmbda[self.positive_labels] = self.lmbda_upper_bound

            if isclose(self.lmbda_upper_bound, self.q_hat):
                if self.config.cf_opt.use_group_information:
                    self.opt_lmbda[:, self.positive_labels] = self.q_hat
                else:
                    self.opt_lmbda[self.positive_labels] = self.q_hat
                # Early stop label is now 1D in both modes
                self.early_stop_label[self.positive_labels] = True

            self.lambdas = torch.linspace(
                self.q_hat, self.lmbda_upper_bound, self.num_rounds - 2
            )

        elif self.stage == FedConfFairStage.POP_STAT:
            p, m = self._pop_stat_aggregate_fit(server_round, results, failures)
        elif self.stage == FedConfFairStage.CF_ITER:
            p, m = self._satisfy_lambda_aggregate_fit(server_round, results, failures)
        else:
            raise NotImplementedError

        return p, m

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Using evluation to send U and L values if needed to a specific client"""
        send_list = []

        return send_list

    def __switch_state(self):
        if self.stage == FedConfFairStage.INIT_STAGE:
            self.stage = FedConfFairStage.FED_CP
        elif self.stage == FedConfFairStage.FED_CP:
            self.stage = FedConfFairStage.POP_STAT
        elif self.stage == FedConfFairStage.POP_STAT:
            self.stage = FedConfFairStage.CF_ITER
        else:
            self.cf_round += 1

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        # This the last function called in a round - so state change happens here
        # self.__switch_state()
        # We now moved it to the start of configure fit since that will happen each time
        return None, dict()

    def evaluate(
        self, server_round: int, parameters: Parameters
    ) -> Optional[tuple[float, dict[str, Scalar]]]:
        """Evaluate the current model parameters."""

        if self.test_scores is None or server_round <= self.num_rounds - 1:
            # return None
            cg_dict = dict()
            for i, label in enumerate(self.active_pos_label):
                # active_pos_label is now just labels in both modes
                cg_dict[f"Label {label} CG"] = self.active_cg[i]

            cg_dict["CF Stage"] = self.stage
            return 0, cg_dict

        base_prediction_sets = PredSetTransformation(
            threshold=self.q_hat
        ).pipe_transform(self.test_scores)

        base_cov = calc_coverage(
            prediction_sets=base_prediction_sets, labels=self.test_y
        )
        base_eff = calc_efficiency(prediction_sets=base_prediction_sets)

        # Apply group-aware or standard thresholds
        if self.config.cf_opt.use_group_information:
            # Use full 2D threshold matrix indexed by group: shape (n, num_classes)
            prediction_sets = PredSetTransformation(
                threshold=self.opt_lmbda[self.test_sens]
            ).pipe_transform(self.test_scores)
        else:
            prediction_sets = PredSetTransformation(
                threshold=self.opt_lmbda
            ).pipe_transform(self.test_scores)

        cov = calc_coverage(prediction_sets=prediction_sets, labels=self.test_y)
        eff = calc_efficiency(prediction_sets=prediction_sets)

        res = {
            "use_mle": self.config.cf_opt.use_mle,
            "sketch_method": self.quantile_method,
            "base_coverage": base_cov,
            "base_efficiency": base_eff,
            "coverage": cov,
            "efficiency": eff,
        }
        res.update(self.fairness_eval(prediction_sets, base_prediction_sets))
        res["qhat"] = self.q_hat
        if self.config.cf_opt.use_group_information:
            # For group-aware: log all per-group lambdas for labels
            for i, label in enumerate(self.positive_labels):
                for g_id in range(self.num_sens):
                    res[f"lambda_opt_label_{label}_group_{g_id}"] = self.opt_lmbda[
                        g_id, label
                    ]
        else:
            # For non-group: log all lambdas
            res.update(
                {
                    f"lambda_opt_{i}": lmbda
                    for i, lmbda in enumerate(self.opt_lmbda.tolist())
                }
            )

        res.update(
            {
                f"client_formulation_{i}": form
                for i, form in enumerate(self.client_formulations.values())
            }
        )

        for i, (cid, byte_list) in enumerate(self.client_bytes_sent.items()):
            avg_bytes = sum(byte_list) / len(byte_list)
            print(f"Average bytes sent to server for client {i} (cid={cid}): {avg_bytes:.2f}")
            res[f"avg_bytes_sent_client_{i}"] = avg_bytes

        res["use_descent"] = self.config.cf_opt.use_descent_based_approach
        res["num_opt_rounds"] = self.config.cf_opt.num_opt_rounds
        res["use_group_info"] = self.config.cf_opt.use_group_information

        conf_dict, out_dir, out_file = get_fair_configs_res_path(self.config)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        conf_dict.update(res)
        if self.config.cf_opt.use_group_information:
            os.makedirs(os.path.join(out_dir, "group_aware"), exist_ok=True)
            total_path = os.path.join(out_dir, "group_aware", out_file)
        else:
            total_path = os.path.join(out_dir, out_file)

        df = pd.DataFrame(conf_dict, index=[0])
        append_row_by_column_name(df, total_path)

        return cov, res

    def _pop_stat_configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        clients = client_manager.all()
        assert (
            self.stage == FedConfFairStage.POP_STAT
        ), "Needs to be in the pop_stat stage"
        config = {
            FedConfFairStage.key: self.stage,
            FedCFConstants.F_M: self.config.fairness_metric,
        }
        # Sending the positive labels as params
        params = ndarrays_to_parameters([self.positive_labels.cpu().detach().numpy()])
        return [(client, FitIns(params, config)) for client in clients.values()]

    def _pop_stat_aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregates the probability of satisfying a fairness metric for each client"""

        assert (
            self.stage == FedConfFairStage.POP_STAT
        ), "Needs to be in the pop_stat stage"

        self.U = torch.zeros((self.num_sens, self.positive_labels.shape[0]))
        self.L = torch.zeros_like(self.U)

        if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
            self.U_2 = torch.zeros_like(self.U)
            self.L_2 = torch.zeros_like(self.L)

        # Aggregating the lower and upper levels of the population stats via the
        # law of total proability - we assume the probability is proportional to the
        # number of cvoariates each client has
        for idx, res in enumerate(results):
            gamma = (res[1].num_examples + 1) / (self.N + self.K)
            # Assume only one array is sent. For Equalized Odds assume the first one is equal. op.
            lower_cov = torch.from_numpy(parameters_to_ndarrays(res[1].parameters)[0])
            upper_cov = lower_cov + (1 / (res[1].num_examples + 1))
            self.U += gamma * upper_cov
            self.L += gamma * lower_cov

            if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                # Assume the second ndarray corresponds to the second F_m (i.e pred. equ.)
                lower_cov = torch.from_numpy(
                    parameters_to_ndarrays(res[1].parameters)[1]
                )
                upper_cov = lower_cov + (1 / (res[1].num_examples + 1))

                self.U_2 += gamma * upper_cov
                self.L_2 += gamma * lower_cov

            self.client_formulations[res[0].cid] = res[1].metrics[
                FedCFConstants.CLIENT_FORM
            ]

        self.all_low_overhead_formulation = all(
            form == CommFormulations.LOW_OVERHEAD.value
            for form in self.client_formulations.values()
        )

        # initializing lambdas as the global parameters
        self._snap_lmbda_for_hist_dp()
        init_cf_lambdas = ndarrays_to_parameters(
            [self.curr_lmbda.cpu().detach().numpy()]
        )
        metrics: Dict[str, Scalar] = dict()
        return init_cf_lambdas, metrics

    def _snap_lmbda_for_hist_dp(self):
        """With the one-shot histogram DP release, only query lambdas on the
        client histogram grid (see fed_dp.snap_to_grid)."""
        if self.config.add_dp_noise and self.config.dp_release == "histogram":
            self.curr_lmbda = snap_to_grid(
                self.curr_lmbda, self.config.dp_hist_bins, float(self.lmbda_upper_bound)
            )

    def _satisfy_lambda_configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.
        This is the optimization framework for conformal fairness. It is reworked from
        the original paper to avoid discretization and be more efficient."""
        clients = client_manager.all()
        assert (
            self.stage == FedConfFairStage.CF_ITER
        ), "Needs to be in the CF_ITER stage"
        config = {FedConfFairStage.key: self.stage}
        config["use_group_information"] = self.config.cf_opt.use_group_information

        send_list = []
        # some clients may require the additional parameter
        if self.cf_round == 0:
            nd_arrays = [
                self.U.cpu().detach().numpy(),
                self.L.cpu().detach().numpy(),
            ]
            if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                nd_arrays = [
                    self.U_2.cpu().detach().numpy(),
                    self.L_2.cpu().detach().numpy(),
                ] + nd_arrays

            # Concat the params
            param = ndarrays_to_parameters(
                parameters_to_ndarrays(parameters) + nd_arrays
            )
            clients = client_manager.all()
            # if u,l not being sent

            # if u_l being sent
            config_ul = deepcopy(config)
            config_ul[SpecialParameters.key] = SpecialParameters.UPDATE_U_L

            for client in clients.values():
                if (
                    self.client_formulations[client.cid]
                    == CommFormulations.MORE_PRIVATE.value
                ):
                    send_list.append((client, FitIns(param, config_ul)))
                else:
                    send_list.append((client, FitIns(parameters, config)))

        # update the positive list if needed
        elif self.update_pos_label_client:
            param = ndarrays_to_parameters(
                parameters_to_ndarrays(parameters)
                + [self.active_pos_label.cpu().detach().numpy()]
            )
            config_pos = deepcopy(config)
            config_pos[SpecialParameters.key] = SpecialParameters.UPDATE_POS

            send_list = [
                (client, FitIns(param, config_pos)) for client in clients.values()
            ]
            self.update_pos_label_client = False
        else:
            send_list = [
                (client, FitIns(parameters, config)) for client in clients.values()
            ]

        return send_list

    def _satisfy_lambda_aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Choosing new Lambdas According to the Algorithm"""

        assert (
            self.stage == FedConfFairStage.CF_ITER
        ), "Needs to be in the CF_ITER stage"
        c = self.config.closeness_measure

        use_descent = self.config.cf_opt.use_descent_based_approach

        cg_curr = self._coverage_gap(results)  # cg for each positive label

        if not use_descent:
            self.curr_lmbda = (
                torch.ones_like(self.curr_lmbda, dtype=torch.float)
                * self.lambdas[self.cf_round]
            )

        # Get optimal lambda values for current active labels
        if self.config.cf_opt.use_group_information:
            # For group-aware: extract optimal lambdas as 2D tensor (num_groups, num_active_labels)
            opt_lmbda_curr = torch.zeros((self.num_sens, len(self.active_pos_label)))
            for g_id in range(self.num_sens):
                opt_lmbda_curr[g_id] = self.opt_lmbda[g_id, self.active_pos_label]
        else:
            opt_lmbda_curr = self.opt_lmbda[self.active_pos_label]

        # Mask determining which positive label's lambda needs to be updated since a new optimal is found
        if self.config.cf_opt.use_group_information:
            # For group-aware: mask is 2D (num_groups, num_active_labels), one per (group, label) pair
            # Broadcast cg_curr from (num_active_labels,) to (num_groups, num_active_labels)
            cg_broadcast = cg_curr.unsqueeze(0).expand(self.num_sens, -1)
            update_mask_2d = (
                (cg_broadcast < c)
                & (self.curr_lmbda < opt_lmbda_curr)
                & (self.q_hat <= self.curr_lmbda)
            )
            update_mask = torch.all(
                update_mask_2d, dim=0
            )  # (num_active_labels, ) - only update if all groups for a label need updating
            self.opt_lmbda[:, self.active_pos_label[update_mask]] = self.curr_lmbda[
                :, update_mask
            ]
        else:
            update_mask = (
                (cg_curr < c)
                & (self.curr_lmbda < opt_lmbda_curr)
                & (self.q_hat <= self.curr_lmbda)
            )
            self.opt_lmbda[self.active_pos_label[update_mask]] = self.curr_lmbda[
                update_mask
            ]
        lr = -1
        if use_descent:
            if self.cf_round == 0 and torch.any(update_mask):
                # converged so no need to continue updating
                self.active_pos_label = self.active_pos_label[~update_mask]
                cg_curr = cg_curr[~update_mask]

                if self.config.cf_opt.use_group_information:
                    self.b_curr = self.b_curr[:, ~update_mask]
                    self.curr_lmbda = self.curr_lmbda[:, ~update_mask]
                else:
                    self.b_curr = self.b_curr[~update_mask]
                    self.curr_lmbda = self.curr_lmbda[~update_mask]

                # No need to store the values for coverged labels
                self.U = self.U[:, ~update_mask]
                self.L = self.L[:, ~update_mask]
                if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                    self.U_2 = self.U_2[:, ~update_mask]
                    self.L_2 = self.L_2[:, ~update_mask]

                # effectively all the values here are 0
                update_mask = update_mask[~update_mask]
                # Let the server know it needs to update the positive labels
                self.update_pos_label_client = True
                # the updated ones are non longer active

            # Step direction term based on SGD with momentum
            self.b_curr = self.config.cf_opt.momentum * self.b_curr + (cg_curr - c)

            # in this case (restart mask), since lr > 0 we wont select a lambda in the valid range
            # Options to correct this -> randomly select a new value in the valid range to try
            # Another option is to restart the value of b_curr with cg_curr
            # First option provides better exploration of solutions
            restart_mask = update_mask & (
                self.b_curr > 0
            )  # (num_groups, num_active_labels) or (num_active_labels,) depending on if group info is used

            # Option 1

            # Recompute opt_lmbda_curr with new active_pos_label size
            if self.config.cf_opt.use_group_information:
                opt_lmbda_curr = torch.zeros(
                    (self.num_sens, len(self.active_pos_label))
                )
                for g_id in range(self.num_sens):
                    opt_lmbda_curr[g_id] = self.opt_lmbda[g_id, self.active_pos_label]
            else:
                opt_lmbda_curr = self.opt_lmbda[self.active_pos_label]

            lmbda_diff = torch.where(
                self.b_curr > 0,
                opt_lmbda_curr - self.curr_lmbda,
                self.q_hat - self.curr_lmbda,
            )

            lr = self._update_learning_rate(
                self.config.cf_opt.lr,
                self.b_curr[~restart_mask],
                lmbda_diff[~restart_mask],
            )

            self.b_curr[restart_mask] = 0  # Since we are restarting these values

            next_lmbda = torch.zeros_like(restart_mask, dtype=torch.float)
            next_lmbda[~restart_mask] = (
                self.curr_lmbda[~restart_mask] + lr * self.b_curr[~restart_mask]
            )

            if restart_mask.any():
                if self.config.cf_opt.use_group_information and (
                    server_round < self.num_rounds // 2
                ):
                    # next_lmbda[restart_mask] = (
                    #     opt_lmbda_curr[restart_mask] - self.q_hat
                    # ) * torch.rand(
                    #     torch.sum(restart_mask[0]).item(), dtype=torch.float
                    # ) .unsqueeze(
                    #     0
                    # ).expand(
                    #     self.num_sens, -1
                    # ) + self.q_hat  # Random restart within [q_hat, opt_lmbda] ignoring group information for the first half
                    for i in range(self.num_sens):
                        mask_row = restart_mask[i]
                        num_true = mask_row.sum().item()

                        rand_vals = torch.rand(num_true)

                        next_lmbda[i, mask_row] = (
                            opt_lmbda_curr[i, mask_row] - self.q_hat
                        ) * rand_vals + self.q_hat
                else:
                    next_lmbda[restart_mask] = (
                        opt_lmbda_curr[restart_mask] - self.q_hat
                    ) * torch.rand(
                        torch.sum(restart_mask).item(),
                        dtype=torch.float,  # This is automatically broadcasted
                    ) + self.q_hat  # Random restart within [q_hat, opt_lmbda]

            # breakpoint()

            # Option 2
            # self.b_curr[restart_mask] = cg_curr - c

            # lmbda_diff = torch.where(self.b_curr > 0, self.opt_lmbda[self.active_pos_label] - self.curr_lmbda[self.active_pos_label],
            #                          self.q_hat-self.curr_lmbda[self.active_pos_label])

            # lr = self._update_learning_rate(self.config.cf_opt.lr, self.b_curr, lmbda_diff)

            # next_lmbda = self.curr_lmbda + lr*self.b_curr

            # Now we can test the next lambda value in the appropriate range
            self.curr_lmbda = next_lmbda

            stop_mask = (self.curr_lmbda >= self.lmbda_upper_bound) | (
                self.curr_lmbda <= self.q_hat
            )
        else:
            stop_mask = update_mask

        # Update early_stop_label for those indices
        # Note: active_pos_label is now just labels in both modes
        if self.config.cf_opt.use_group_information:
            # stop_mask is 2D (num_groups, num_active_labels)
            # A label should stop only if all groups have stopped
            fully_stopped_mask = torch.all(stop_mask, dim=0)
            stopped_labels = self.active_pos_label[fully_stopped_mask]
            self.early_stop_label[stopped_labels] = True
        else:
            # stop_mask is 1D (num_active_labels)
            stopped_labels = self.active_pos_label[stop_mask]
            self.early_stop_label[stopped_labels] = True

        # Prepare parameters for sending to clients
        self._snap_lmbda_for_hist_dp()
        if self.config.cf_opt.use_group_information:
            # Flatten 2D tensor for transmission
            lmbda_to_send = self.curr_lmbda.flatten().cpu().detach().numpy()
        else:
            lmbda_to_send = self.curr_lmbda.cpu().detach().numpy()

        params = ndarrays_to_parameters([lmbda_to_send])
        self.active_cg = cg_curr
        if use_descent:
            logging.info(
                {
                    "cg_curr": cg_curr,
                    "lr_curr": lr,
                    "lmbda_diff": lmbda_diff,
                    "b_curr": self.b_curr,
                    "lambda_opt": self.opt_lmbda,
                    "restart_mask": restart_mask,
                    "update_mask": update_mask,
                    "early_stop_label": self.early_stop_label,
                    "active_pos_label": self.active_pos_label,
                }
            )
        else:
            logging.info(
                {
                    "cg_curr": cg_curr,
                    "lr_curr": lr,
                    "lambda_opt": self.opt_lmbda,
                    "update_mask": update_mask,
                    "early_stop_label": self.early_stop_label,
                    "active_pos_label": self.active_pos_label,
                }
            )
        return params, {
            "cg_curr": cg_curr,
            "lr_curr": lr,
            "b_curr": self.b_curr,
            "lambda_opt": self.opt_lmbda,
        }

    def _coverage_gap(self, results: List[Tuple[ClientProxy, FitRes]]) -> torch.Tensor:
        """Output is of length of active_positive_labels (i.e non-converged pos labels)"""

        """
        If all clients use the low overhead version - then we can compute the maximal coverage
        gap more efficiently.
        """
        if self.all_low_overhead_formulation:
            # num_sens x num_active_pos_labels
            low_cov = torch.zeros_like(self.L, dtype=torch.float)
            up_cov = torch.zeros_like(self.U, dtype=torch.float)

            if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                low_cov_2 = torch.zeros_like(self.L_2, dtype=torch.float)
                up_cov_2 = torch.zeros_like(self.U_2, dtype=torch.float)

            sum_gamma = 0
            for idx, res in enumerate(results):
                p_list = parameters_to_ndarrays(
                    res[1].parameters
                )  # lower, upper, lower_2, upper 2
                gamma = (res[1].num_examples + 1) / (self.N + self.K)

                low_cov += gamma * torch.from_numpy(p_list[0]) / self.U
                up_cov += gamma * torch.from_numpy(p_list[1]) / self.L

                sum_gamma += gamma
                if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                    low_cov_2 += gamma * torch.from_numpy(p_list[2]) / self.U_2
                    up_cov_2 += gamma * torch.from_numpy(p_list[3]) / self.L_2

            up_cov = up_cov.minimum(torch.tensor(1))

            cg = (
                up_cov.max(dim=0)[0] - low_cov.min(dim=0)[0]
            )  # Returns cg for each positive label
            if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                cg_2 = up_cov_2.max(dim=0)[0] - low_cov_2.min(dim=0)[0]
                # Report the Maximum Converage Gaps
                cg = torch.maximum(cg, cg_2)

            return cg

        # Either all use the more private approach or the hybrid approach
        else:
            pairwise_cg = torch.zeros(
                (self.num_sens, self.num_sens, len(self.active_pos_label))
            )

            if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                pairwise_cg_2 = torch.zeros_like(pairwise_cg)

            for idx, res in enumerate(results):
                gamma = (res[1].num_examples + 1) / (self.N + self.K)
                p_list = parameters_to_ndarrays(res[1].parameters)

                if (
                    self.client_formulations[res[0].cid]
                    == CommFormulations.MORE_PRIVATE.value
                ):
                    pairwise_cg += gamma * torch.from_numpy(p_list[0])

                    if (
                        self.config.fairness_metric
                        == FairnessMetric.EQUALIZED_ODDS.value
                    ):
                        pairwise_cg_2 += gamma * torch.from_numpy(p_list[1])

                elif (
                    self.client_formulations[res[0].cid]
                    == CommFormulations.LOW_OVERHEAD.value
                ):
                    low_cov = torch.from_numpy(p_list[0]) / self.U
                    up_cov = torch.from_numpy(p_list[1]) / self.L

                    pairwise_cg += gamma * (up_cov[:, None, :] - low_cov[None, :, :])

                    if (
                        self.config.fairness_metric
                        == FairnessMetric.EQUALIZED_ODDS.value
                    ):
                        low_cov_2 = torch.from_numpy(p_list[2]) / self.U_2
                        up_cov_2 = torch.from_numpy(p_list[3]) / self.L_2
                        pairwise_cg_2 += gamma * (
                            up_cov_2[:, None, :] - low_cov_2[None, :, :]
                        )

            cg = pairwise_cg.amax(dim=(0, 1))
            if self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                cg_2 = pairwise_cg_2.amax(dim=(0, 1))
                cg = torch.maximum(cg, cg_2)

            return cg

    def _update_learning_rate(
        self,
        initial_lr: float,
        b_curr: torch.Tensor,
        lmbda_diff: torch.Tensor,
        eps=1e-12,
    ) -> torch.Tensor:
        b_safe = torch.where(b_curr.abs() < eps, torch.full_like(b_curr, eps), b_curr)

        # Max allowed step size to stay within range
        a_max = lmbda_diff / b_safe

        # Clamp a_max to valid range
        a_max = a_max.clamp(min=eps, max=initial_lr)

        # Compute how many halvings needed
        ratio = initial_lr / a_max
        l = torch.ceil(torch.log2(ratio)).to(dtype=torch.int)
        l = torch.where(l > 0, l, torch.zeros_like(l))

        # Apply halving and clamp to not overshoot
        a_t = initial_lr * torch.pow(0.5, l)
        a_t = torch.min(a_t, a_max)

        return a_t

    def fairness_eval(self, pred_sets, baseline_pred_sets):
        """Adapted from the original conformal fairness implementation"""
        res = dict()
        res["violation"] = 0
        res["base_violation"] = 0

        for label in torch.arange(self.test_dm.num_classes):
            filtered_test_mask = self.test_y == label

            cov_labels = self.test_y[filtered_test_mask]

            cov = calc_coverage(pred_sets[filtered_test_mask, :], cov_labels)

            base_cov = calc_coverage(
                baseline_pred_sets[filtered_test_mask, :], cov_labels
            )

            print(f"Base Class-Conditional Coverage for y_k = {label} = {base_cov}")
            print(f"Class-Conditional Coverage for y_k = {label} = {cov}")

            res[f"class_cond_cov_y={label}"] = cov
            res[f"base_class_cond_cov_y={label}"] = base_cov

        print("\n\n")

        for group in torch.arange(self.test_dm.num_sensitive_groups):
            filtered_test_mask = self.test_sens == group

            cov_labels = self.test_y[filtered_test_mask]

            cov = calc_coverage(pred_sets[filtered_test_mask, :], cov_labels)

            base_cov = calc_coverage(
                baseline_pred_sets[filtered_test_mask, :], cov_labels
            )

            print(f"Base Group-Conditional Coverage for g_i = {group} = {base_cov}")
            print(f"Group-Conditional Coverage for g_i = {group} = {cov}")

            res[f"group_cond_cov_g={group}"] = cov
            res[f"base_group_cond_cov_g={group}"] = base_cov

        print("\n\n")

        for label in torch.arange(self.test_dm.num_classes):
            coverages = []
            baseline_coverages = []
            for g_i in range(self.test_dm.num_sensitive_groups):
                match self.config.fairness_metric:
                    case (
                        FairnessMetric.EQUAL_OPPORTUNITY.value
                        | FairnessMetric.PREDICTIVE_EQUALITY.value
                        | FairnessMetric.EQUALIZED_ODDS.value
                        | FairnessMetric.DEMOGRAPHIC_PARITY.value
                        | FairnessMetric.DISPARATE_IMPACT.value
                        | FairnessMetric.OVERALL_ACC_EQUALITY.value
                    ):
                        filtered_test_mask = get_filter_mask(
                            fairness_metric=self.config.fairness_metric,
                            labels=self.test_y,
                            groups=self.test_sens,
                            pos_label=label,
                            group_id=g_i,
                        )

                        if isinstance(filtered_test_mask, Tuple):
                            temp_cov = []
                            temp_base_cov = []
                            component = "eo"
                            for mask in filtered_test_mask:
                                cov_labels = torch.full_like(self.test_y[mask], label)
                                cov = calc_coverage(pred_sets[mask, :], cov_labels)
                                temp_cov.append(cov)

                                base_cov = calc_coverage(
                                    baseline_pred_sets[mask, :], cov_labels
                                )
                                temp_base_cov.append(base_cov)

                                component = "pe"
                                res[f"pos_cov_{component}_y={label}_g={g_i}"] = cov
                                res[f"base_pos_cov_{component}_y={label}_g={g_i}"] = (
                                    base_cov
                                )

                            coverages.append(temp_cov)
                            baseline_coverages.append(temp_base_cov)
                        else:
                            if (
                                self.config.fairness_metric
                                != FairnessMetric.OVERALL_ACC_EQUALITY.value
                            ):
                                cov_labels = torch.full_like(
                                    self.test_y[filtered_test_mask], label
                                )
                            else:
                                cov_labels = self.test_y[filtered_test_mask]

                            cov = calc_coverage(
                                pred_sets[filtered_test_mask, :], cov_labels
                            )
                            coverages.append(cov)

                            base_cov = calc_coverage(
                                baseline_pred_sets[filtered_test_mask, :], cov_labels
                            )
                            baseline_coverages.append(base_cov)

                            print(
                                f"Positive Label Coverage for y_k = {label} and g_i = {g_i} = {cov}"
                            )

                            res[f"pos_cov_y={label}_g={g_i}"] = cov
                            res[f"base_pos_cov_y={label}_g={g_i}"] = base_cov

                    case FairnessMetric.PREDICTIVE_PARITY.value:
                        try:
                            eo_filtered_test_mask = get_filter_mask(
                                fairness_metric=FairnessMetric.EQUAL_OPPORTUNITY.value,
                                labels=self.test_y,
                                groups=self.test_sens,
                                pos_label=label,
                                group_id=g_i,
                            )

                            dp_filtered_test_mask = get_filter_mask(
                                fairness_metric=FairnessMetric.DEMOGRAPHIC_PARITY.value,
                                labels=self.test_y,
                                groups=self.test_sens,
                                pos_label=label,
                                group_id=g_i,
                            )
                        finally:
                            pass

                        pos_labels = torch.full_like(
                            self.test_y[eo_filtered_test_mask], label
                        )
                        eo_coverage = calc_coverage(
                            pred_sets[eo_filtered_test_mask, :], pos_labels
                        )
                        eo_base_coverage = calc_coverage(
                            baseline_pred_sets[eo_filtered_test_mask, :], pos_labels
                        )
                        pos_labels = torch.full_like(
                            self.test_y[dp_filtered_test_mask], label
                        )
                        dp_coverage = calc_coverage(
                            pred_sets[dp_filtered_test_mask, :], pos_labels
                        )
                        dp_base_coverage = calc_coverage(
                            baseline_pred_sets[dp_filtered_test_mask, :], pos_labels
                        )

                        prior = (
                            self.test_y[dp_filtered_test_mask] == pos_labels
                        ).sum() / (dp_filtered_test_mask.sum())

                        cov = abs(eo_coverage * prior / dp_coverage - prior)
                        coverages.append(cov)

                        base_cov = abs(
                            eo_base_coverage * prior / dp_base_coverage - prior
                        )
                        baseline_coverages.append(base_cov)

                        print(
                            f"Positive Label Coverage for y_k = {label} and g_i = {g_i} = {cov}"
                        )

                        res[f"pos_cov_y={label}_g={g_i}"] = cov

            if label not in self.positive_labels:
                continue  # Skip non-positive labels for violation calculations

            if self.config.fairness_metric == FairnessMetric.DISPARATE_IMPACT.value:
                cov_ratio = min(coverages) / max(coverages)
                base_cov_ratio = min(baseline_coverages) / max(baseline_coverages)
                print(f"Actual Coverage Ratio={cov_ratio}\n")
                res["violation"] = min(res["violation"], cov_ratio)
                res["base_violation"] = min(res["base_violation"], base_cov_ratio)
                res[f"violation_label={label}"] = cov_ratio
                res[f"base_violation_label={label}"] = base_cov_ratio
            elif self.config.fairness_metric == FairnessMetric.EQUALIZED_ODDS.value:
                cov_delta_eo = max([x[0] for x in coverages]) - min(
                    [x[0] for x in coverages]
                )
                cov_delta_pe = max([x[1] for x in coverages]) - min(
                    [x[1] for x in coverages]
                )

                print(f"Actual Coverage Delta = {max(cov_delta_eo, cov_delta_pe)}\n")
                res[f"violation_eo_label={label}"] = cov_delta_eo
                res[f"violation_pe_label={label}"] = cov_delta_pe

                res["violation"] = max(
                    max(
                        (
                            cov_delta_eo,
                            cov_delta_pe,
                        )
                    ),
                    res["violation"],
                )

                base_cov_delta_eo = max([x[0] for x in baseline_coverages]) - min(
                    [x[0] for x in baseline_coverages]
                )
                base_cov_delta_pe = max([x[1] for x in baseline_coverages]) - min(
                    [x[1] for x in baseline_coverages]
                )

                res[f"base_violation_eo_label={label}"] = base_cov_delta_eo
                res[f"base_violation_pe_label={label}"] = base_cov_delta_pe

                res["base_violation"] = max(
                    max(
                        (
                            base_cov_delta_eo,
                            base_cov_delta_pe,
                        )
                    ),
                    res["base_violation"],
                )
            else:
                cov_delta = max(coverages) - min(coverages)
                base_cov_delta = max(baseline_coverages) - min(baseline_coverages)
                print(f"Actual Coverage Delta={cov_delta}\n")
                res["violation"] = max(cov_delta, res["violation"])
                res["base_violation"] = max(base_cov_delta, res["base_violation"])
                res[f"violation_label={label}"] = cov_delta
                res[f"base_violation_label={label}"] = base_cov_delta

        return res
