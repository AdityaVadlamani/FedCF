import os
import torch
from typing import Dict

from flwr.common import Context, ndarrays_to_parameters, Parameters
from flwr.server import ServerAppComponents, ServerConfig
import xgboost as xgb


xgb.set_config(verbosity=0)

from fairlex_datamodule import FairlexDatamodule
from pokec_datamodule import PokecDataModule
from task import (
    get_gnn_model,
    test,
    set_weights,
    get_weights,
    get_model,
    test_gnn,
    get_fairlex_model,
)
from strategy import CustomFedAvg, CustomFedXgbBagging
from fed_config import FedBaseExptConfig
from folktables_datamodule import FolktablesDataModule
from fitzpatrick_datamodule import FitzpatrickDataModule

from conformal_fairness.constants import Stage
from conformal_fairness.data import BaseDataModule


def server_fn(context: Context, args: FedBaseExptConfig, datamodule: BaseDataModule):
    # Read from config
    num_rounds = args.num_server_rounds
    fraction_fit = args.fraction_fit

    if isinstance(
        datamodule, FolktablesDataModule
    ):  # Uses XGBoost specific strategy, evaluation, etc.

        def config_func(rnd: int) -> Dict[str, str]:
            """Return a configuration with global epochs."""
            config = {
                "global_round": str(rnd),
            }
            return config

        def gen_xgb_evaluate_fn(datamodule: BaseDataModule, params):
            valid_labels = (
                datamodule.y[datamodule.split_dict[Stage.VALIDATION]]
                .cpu()
                .detach()
                .numpy()
            )
            valid_features = (
                datamodule.X[datamodule.split_dict[Stage.VALIDATION], :]
                .cpu()
                .detach()
                .numpy()
            )

            valid_dmatrix = xgb.DMatrix(valid_features, valid_labels)

            def evaluate(server_round: int, parameters, _config):
                if server_round == 0:
                    return 0, {}
                # Load global model
                bst = xgb.Booster(params=params)
                for para in parameters.tensors:
                    para_b = bytearray(para)
                bst.load_model(para_b)

                # Run evaluation
                eval_results = bst.eval_set(
                    evals=[(valid_dmatrix, "valid")],
                    iteration=bst.num_boosted_rounds() - 1,
                )
                error = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

                return 0.0, {"accuracy": 1 - error}

            return evaluate

        def evaluate_metrics_aggregation(eval_metrics):
            """Return an aggregated metric for evaluation."""
            total_num = sum([num for num, _ in eval_metrics])
            aggregated = (
                sum([metrics["accuracy"] * num for num, metrics in eval_metrics])
                / total_num
            )
            metrics_aggregated = {"accuracy": aggregated}
            return metrics_aggregated

        params_dict = {
            "tree_method": args.base_model_config.tree_method,
            "objective": "multi:softprob",
            "num_class": datamodule.num_classes,
            "eval_metric": "merror",
            "max_depth": args.base_model_config.max_depth,
            "max_leaves": args.base_model_config.max_leaves,
            "grow_policy": args.base_model_config.grow_policy,
            "booster": args.base_model_config.booster,
            "gamma": args.base_model_config.gamma,
            "colsample_bytree": args.base_model_config.colsample_bytree,
            "colsample_bylevel": args.base_model_config.colsample_bylevel,
            "colsample_bynode": args.base_model_config.colsample_bynode,
            "subsample": args.base_model_config.subsample,
            "reg_alpha": args.base_model_config.reg_alpha,
            "reg_lambda": args.base_model_config.reg_lambda,
        }

        parameters = Parameters(tensor_type="", tensors=[])
        strategy = CustomFedXgbBagging(
            save_root=os.path.join(args.output_dir, args.job_id),
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            evaluate_function=gen_xgb_evaluate_fn(
                datamodule=datamodule, params=params_dict
            ),
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation,
            on_evaluate_config_fn=config_func,
            on_fit_config_fn=config_func,
            initial_parameters=parameters,
        )
    else:

        if isinstance(datamodule, PokecDataModule):
            model = get_gnn_model(
                config=args.base_model_config,
                num_features=datamodule.num_features,
                num_classes=datamodule.num_classes,
            )
        elif isinstance(datamodule, FitzpatrickDataModule):
            model = get_model(
                args.base_model_config.architecture, datamodule.num_classes
            )
        elif isinstance(datamodule, FairlexDatamodule):
            model = get_fairlex_model(args.dataset.name, datamodule.num_classes)
        else:
            raise NotImplementedError(
                f"Datamodule {type(datamodule)} not implemented for server with NN."
            )

        # Initialize model parameters
        ndarrays = get_weights(model)
        parameters = ndarrays_to_parameters(ndarrays)

        def gen_evaluate_fn(model_gen_fn, dm: BaseDataModule):
            """Return a Flower `evaluate_fn` that runs on the server's held-out test set."""
            testloader = dm.val_dataloader()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            def evaluate(server_round: int, parameters, _config):
                model = model_gen_fn()
                set_weights(model, parameters)
                if dm.name == "Pokec":
                    loss, acc = test_gnn(model, testloader, device)
                else:
                    loss, acc = test(model, testloader, device)
                # Flower expects (loss, metrics_dict)
                return loss, {"centralized_accuracy": acc}

            return evaluate

        if isinstance(datamodule, FitzpatrickDataModule):
            model_gen_fn = lambda: get_model(
                args.base_model_config.architecture, datamodule.num_classes
            )
        elif isinstance(datamodule, PokecDataModule):
            model_gen_fn = lambda: get_gnn_model(
                config=args.base_model_config,
                num_features=datamodule.num_features,
                num_classes=datamodule.num_classes,
            )
        elif isinstance(datamodule, FairlexDatamodule):
            model_gen_fn = lambda: get_fairlex_model(
                args.dataset.name, datamodule.num_classes
            )
        else:
            raise NotImplementedError(
                f"Datamodule {type(datamodule)} not implemented for server with NN."
            )

        evaluate_fn = gen_evaluate_fn(model_gen_fn, datamodule)

        # Define strategy
        strategy = CustomFedAvg(
            save_root=os.path.join(args.output_dir, args.job_id),
            model_gen_fn=model_gen_fn,
            num_classes=datamodule.num_classes,
            fraction_fit=fraction_fit,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            evaluate_fn=evaluate_fn,
        )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)
