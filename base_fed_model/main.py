import itertools
import os
import re
from pathlib import Path
import warnings

from transformers import PreTrainedModel


warnings.filterwarnings("ignore", category=UserWarning, module="outdated")

from conformal_fairness.constants import CAIL

from fairlex_datamodule import FairlexDatamodule
from fed_constants import FOLKTABLES_OPTIONS
from fed_config import FedBaseExptConfig
from fitzpatrick_datamodule import FitzpatrickDataModule
from folktables_datamodule import FolktablesDataModule
from pokec_datamodule import PokecDataModule

import torch
import xgboost as xgb

xgb.set_config(verbosity=0)  # Disable warnings

from client_app import client_fn
from server_app import server_fn
from task import (
    get_fairlex_model,
    get_gnn_model,
    get_model,
    run_inference_alldl,
    run_xgb_inference_alldl,
    output_results,
)

from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp
import pyrallis.argparsing as pyr_a
from lightning import seed_everything

from conformal_fairness.constants import ACS_INCOME, ACS_EDUC, FITZPATRICK, POKEC


def clear_output_dir(base_dir, job_id):
    folder_path = os.path.join(base_dir, job_id)
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isdir(item_path):
                for root, dirs, files in os.walk(item_path, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(item_path)
            else:
                os.remove(item_path)


def get_best_checkpoint(base_dir, job_id):
    base_path = Path(base_dir, job_id)

    # Patterns to match checkpoint files
    checkpoint_file_patterns = [
        base_path.rglob("model_state_acc_*_round_*.pt"),
        base_path.rglob("model_state_acc_*_round_*.json"),
    ]

    # Collect all matching checkpoint files
    checkpoint_paths = list(itertools.chain(*checkpoint_file_patterns))

    # Regex to match directories named like model_state_acc_*_round_<int>
    dir_name_pattern = re.compile(r"model_state_acc_([0-9.]+)_round_\d+$")

    # Add matching directories to checkpoint paths
    checkpoint_paths.extend(
        p
        for p in base_path.rglob("*")
        if p.is_dir() and dir_name_pattern.fullmatch(p.name)
    )

    acc_re = re.compile(r"model_state_acc_([0-9.]+)_round_")

    best_ckpt = None
    best_acc = -float("inf")
    best_time = -1

    for p in checkpoint_paths:
        m = acc_re.search(p.name)
        if not m:
            continue

        acc = float(m.group(1))
        mtime = p.stat().st_mtime

        if acc > best_acc or (acc == best_acc and mtime > best_time):
            best_acc, best_time, best_ckpt = acc, mtime, p

    return best_ckpt


def main():
    args = pyr_a.parse(config_class=FedBaseExptConfig)
    seed_everything(args.seed)
    print(args)

    if args.dataset.name in (ACS_INCOME, ACS_EDUC):
        if args.folktables_partition_type not in FOLKTABLES_OPTIONS:
            raise ValueError(
                f"Need partition_type to be in {FOLKTABLES_OPTIONS} depending on the partitioning scheme for the US states"
            )
        fdm = FolktablesDataModule(args, partition_type=args.folktables_partition_type)
    elif args.dataset.name == FITZPATRICK:
        fdm = FitzpatrickDataModule(args)
    elif args.dataset.name == POKEC:
        fdm = PokecDataModule(args)
        fdm.setup_sampler(args.base_model_config.layers)
        args.num_clients = 2  # Pokec only has two clients
    elif args.dataset.name in (CAIL):
        fdm = FairlexDatamodule(args)
    else:
        raise ValueError("Invalid dataset provided")

    fdm.prepare_data()
    assert (
        args.dataset_split_fractions is not None
    ), "Dataset split fractions must be provided"

    fdm.setup(args)

    # Construct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=lambda context: client_fn(context, args, fdm))
    print("Client App Created")

    # Create your ServerApp passing the server generation function
    server_app = ServerApp(server_fn=lambda context: server_fn(context, args, fdm))
    print("Server App Created")

    # Build client_resources dict for Flower
    client_resources = {
        "num_cpus": float(args.resource_config.cpus),
        "num_gpus": float(args.resource_config.gpus),
    }

    clear_output_dir(args.output_dir, args.job_id)

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        # equivalent to setting `num-supernodes` in the pyproject.toml
        num_supernodes=args.num_clients,
        backend_config={"client_resources": client_resources},
    )

    best_ckpt = get_best_checkpoint(args.output_dir, args.job_id)

    assert best_ckpt is not None, "No checkpoint files found!"

    if args.dataset.name in (ACS_INCOME, ACS_EDUC):
        model = xgb.Booster(
            {
                "tree_method": args.base_model_config.tree_method,
                "objective": "multi:softprob",
                "num_class": fdm.num_classes,
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
        )
        model.load_model(best_ckpt)

        # run inference:
        print("Model is Loading Properly")
        results_dict = run_xgb_inference_alldl(model, fdm)
    else:
        if args.dataset.name == FITZPATRICK:
            model = get_model(args.base_model_config.architecture, fdm.num_classes)
        elif args.dataset.name == POKEC:
            model = get_gnn_model(
                args.base_model_config, fdm.num_features, fdm.num_classes
            )
        elif args.dataset.name in (CAIL,):
            model = get_fairlex_model(args.dataset.name, fdm.num_classes)
        else:
            raise ValueError("Invalid dataset provided")

        if args.dataset.name in (CAIL,):
            assert isinstance(model, PreTrainedModel)
            model.from_pretrained(best_ckpt)
        else:
            model.load_state_dict(
                torch.load(best_ckpt, map_location="cpu", weights_only=False)
            )

        # run inference:
        print("Model is Loading Properly")
        results_dict = run_inference_alldl(model, fdm)

    print("Writing results to file")
    output_results(args, results_dict, Path(args.output_dir, args.job_id))


if __name__ == "__main__":
    main()
