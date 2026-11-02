from pathlib import Path
import json
from logging import INFO

import torch
from flwr.server.strategy import FedAvg, FedXgbBagging
from flwr.common import logger, parameters_to_ndarrays
from transformers import PreTrainedModel
import xgboost as xgb

from task import set_weights


class CustomFedAvg(FedAvg):
    """
    FedAvg extension for PyTorch:
      • keeps a per‑run output directory
      • records every evaluation in results.json
      • saves the state‑dict of the best global model seen so far
    """

    def __init__(
        self,
        model_gen_fn,
        num_classes: int,
        save_root: Path | str = "outputs",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_path = Path(save_root)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.model_gen_fn = model_gen_fn
        self.num_classes = num_classes
        self.best_acc_so_far: float = 0.0
        self.results: dict[str, list[dict]] = {}

    def _store_results(self, tag: str, result_dict: dict) -> None:
        self.results.setdefault(tag, []).append(result_dict)
        with (self.save_path / "results.json").open("w", encoding="utf-8") as fp:
            json.dump(self.results, fp, indent=2)

    def _save_best(self, rnd: int, accuracy: float, parameters) -> None:
        if accuracy <= self.best_acc_so_far:
            return  # not a new best

        self.best_acc_so_far = accuracy
        logger.log(INFO, "💡 New best global model found: %.4f", accuracy)

        model = self.model_gen_fn()
        set_weights(model, parameters_to_ndarrays(parameters))
        if isinstance(model, PreTrainedModel):
            model.save_pretrained(
                self.save_path / f"model_state_acc_{accuracy:.4f}_round_{rnd}"
            )
        else:
            torch.save(
                model.state_dict(),
                self.save_path / f"model_state_acc_{accuracy:.4f}_round_{rnd}.pt",
            )

    def evaluate(self, server_round: int, parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        acc = metrics.get("centralized_accuracy")
        if acc is not None:
            self._save_best(server_round, acc, parameters)

        self._store_results(
            "centralized_evaluate", {"round": server_round, "loss": loss, **metrics}
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self._store_results(
            "federated_evaluate", {"round": server_round, "loss": loss, **metrics}
        )
        return loss, metrics


class CustomFedXgbBagging(FedXgbBagging):
    """
    FedAvg extension for PyTorch:
      • keeps a per-run output directory
      • records every evaluation in results.json
      • saves the state-dict of the best global model seen so far
    """

    def __init__(
        self,
        save_root: Path | str = "outputs",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.save_path = Path(save_root)
        self.save_path.mkdir(parents=True, exist_ok=True)

        self.best_acc_so_far: float = 0.0
        self.results: dict[str, list[dict]] = {}

    def _store_results(self, tag: str, result_dict: dict) -> None:
        self.results.setdefault(tag, []).append(result_dict)
        with (self.save_path / "results.json").open("w", encoding="utf-8") as fp:
            json.dump(self.results, fp, indent=2)

    def _save_best(self, rnd: int, accuracy: float, parameters) -> None:
        if accuracy <= self.best_acc_so_far:
            return  # not a new best

        self.best_acc_so_far = accuracy
        logger.log(INFO, "💡 New best global model found: %.4f", accuracy)

        bst = xgb.Booster()
        best_model = bytearray(parameters.tensors[0])
        bst.load_model(best_model)
        bst.save_model(
            self.save_path / f"model_state_acc_{accuracy:.4f}_round_{rnd}.json"
        )

    def evaluate(self, server_round: int, parameters):
        loss, metrics = super().evaluate(server_round, parameters)

        acc = metrics.get("accuracy")
        if acc is not None:
            self._save_best(server_round, acc, parameters)

        self._store_results(
            "centralized_evaluate", {"round": server_round, "loss": loss, **metrics}
        )
        return loss, metrics

    def aggregate_evaluate(self, server_round, results, failures):
        loss, metrics = super().aggregate_evaluate(server_round, results, failures)

        self._store_results(
            "federated_evaluate", {"round": server_round, "loss": loss, **metrics}
        )
        return loss, metrics
