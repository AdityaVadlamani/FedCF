import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="outdated")

from flwr.simulation import run_simulation
from flwr.client import ClientApp
from flwr.server import ServerApp
import lightning as L
import pyrallis.argparsing as pyr_a
import torch
import xgboost as xgb

xgb.set_config(verbosity=0)  # Disable warnings


from conformal_fairness.constants import (
    ACS_EDUC,
    ACS_INCOME,
    CAIL,
    LABELS_KEY,
    PROBS_KEY,
    PARTITION_FIELD,
    ConformalMethod,
)

from cp_fed_client_app import fair_client_fn
from cp_server_app import server_fn
from fairlex_datamodule import FairlexDatamodule
from fed_constants import FOLKTABLES_OPTIONS
from fed_config import FedConfFairExptConfig
from fed_utils import apply_distribution_shift, get_test_dm_probs
from fitzpatrick_datamodule import FitzpatrickDataModule
from folktables_datamodule import FolktablesDataModule
from pokec_datamodule import PokecDataModule


def main():
    args = pyr_a.parse(config_class=FedConfFairExptConfig)

    if args.calib_test_equal:  # True by default
        print("Making calib and test set sizes equal", flush=True)
        args.dataset_split_fractions.calib = (
            1
            - (args.dataset_split_fractions.train + args.dataset_split_fractions.valid)
        ) / 2

    if not args.debug_mode:
        sys.breakpointhook = lambda *args, **kwargs: None
        # print("Breakpoints Disabled")
    # else:
    # print("Breakpoints Enabled")

    if args.num_clients == 1:
        assert (
            args.quantile_method == "mean"
        ), "For 1 client (fully centralized), mean should be used."

    probs_labels = torch.load(
        os.path.join(args.output_dir, args.base_job_id, "all_prob_labels.pt"),
        weights_only=False,
    )
    probs = probs_labels[PROBS_KEY]
    L.seed_everything(args.seed)
    print(args, flush=True)

    if args.dataset.name in (ACS_INCOME, ACS_EDUC):
        if args.folktables_partition_type not in FOLKTABLES_OPTIONS:
            raise ValueError(
                f"Need folktables_partition_type to be in {FOLKTABLES_OPTIONS} depending on the partitioning scheme for the US states"
            )
        fdm = FolktablesDataModule(args, partition_type=args.folktables_partition_type)
    elif args.dataset.name == "Fitzpatrick":
        fdm = FitzpatrickDataModule(args)
    elif args.dataset.name == "Pokec":
        fdm = PokecDataModule(args)
    elif args.dataset.name in (CAIL,):
        fdm = FairlexDatamodule(args)
    else:
        raise ValueError("Invalid dataset provided")

    fdm.prepare_data()
    assert (
        args.dataset_split_fractions is not None
    ), "Dataset split fractions must be provided"

    fdm.setup(args)

    fdm.resplit_calib_test(args)

    apply_distribution_shift(fdm, args, probs)

    global_masks = None
    global_client_mapping = None
    if isinstance(fdm, FolktablesDataModule):
        global_masks = fdm.masks
        global_client_mapping = fdm.client_mapping

    labels = probs_labels[LABELS_KEY]
    # Ensuring labels are aligned so that we know the probs are also aligned
    assert torch.all(
        labels == fdm.y
    ), "Definite mismatch between saved and expected labels"

    # Construct the ClientApp passing the client generation function
    client_app = ClientApp(
        client_fn=lambda context: fair_client_fn(
            context,
            args,
            fdm,
            probs,
            args.client_formulations,
            client_mapping=probs_labels.get(PARTITION_FIELD),
            global_masks=global_masks,
            global_client_mapping=global_client_mapping,
        )
    )
    print("Client App Created")

    # Create your ServerApp passing the server generation function
    test_dm, test_probs = get_test_dm_probs(fdm, probs)
    server_app = ServerApp(
        server_fn=lambda context: server_fn(
            context,
            args,
            test_dm,
            test_probs,
            all_probs=(
                probs
                if args.conformal_method.lower() == ConformalMethod.DAPS.value
                else None
            ),
        )
    )
    print("Server App Created")

    # Build client_resources dict for Flower
    client_resources = {
        "num_cpus": float(args.resource_config.cpus),
        "num_gpus": float(args.resource_config.gpus),
    }

    backend_config = {"client_resources": client_resources}
    # On shared (non-exclusive) Slurm nodes, keep Ray inside the job's CPU
    # allocation and give it a per-job temp dir so jobs on the same node
    # don't collide. Opt in with FEDCF_SHARED_NODE=1.
    if os.environ.get("FEDCF_SHARED_NODE") == "1":
        job_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
        if client_resources["num_cpus"] > job_cpus:
            # A client asking for more CPUs than Ray has would never be scheduled.
            print(
                f"resource_config.cpus={client_resources['num_cpus']} exceeds the job's "
                f"{job_cpus} CPUs; capping each client at {job_cpus}"
            )
            client_resources["num_cpus"] = float(job_cpus)
        backend_config["init_args"] = {
            "num_cpus": job_cpus,
            "_temp_dir": f"/tmp/ray_{os.environ.get('SLURM_JOB_ID', os.getpid())}",
            "include_dashboard": False,
        }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        # equivalent to setting `num-supernodes` in the pyproject.toml
        num_supernodes=args.num_clients,
        backend_config=backend_config,
    )


if __name__ == "__main__":
    # python run_fed_cf.py --config_path=fair_fedcf_fitzpatrick_config.yaml
    main()
