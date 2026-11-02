import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="outdated")

import lightning as L
import pyrallis.argparsing as pyr_a
import torch


from conformal_fairness.constants import (
    ACS_EDUC,
    ACS_INCOME,
    LABELS_KEY,
    PROBS_KEY,
    Stage,
)

from fed_constants import FOLKTABLES_OPTIONS
from fed_config import FedConfFairExptConfig
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
    else:
        raise ValueError("Invalid dataset provided")

    fdm.prepare_data()
    assert (
        args.dataset_split_fractions is not None
    ), "Dataset split fractions must be provided"

    fdm.setup(args)

    fdm.resplit_calib_test(args)

    if args.dataset.name in (ACS_INCOME, ACS_EDUC):
        calib_mask = fdm.masks[Stage.CALIBRATION.mask_dstr]
    elif args.dataset.name == "Fitzpatrick":
        calib_mask = fdm._base_dataset.masks[Stage.CALIBRATION.mask_dstr]
    elif args.dataset.name == "Pokec":
        calib_mask = fdm.graph.ndata[Stage.CALIBRATION.mask_dstr]
    else:
        raise ValueError("Invalid dataset provided")

    labels = probs_labels[LABELS_KEY]
    # Ensuring labels are aligned so that we know the probs are also aligned
    assert torch.all(
        labels == fdm.y
    ), "Definite mismatch between saved and expected labels"

    for label in torch.unique(labels):
        if label.item() < 0:
            continue  # Skip invalid labels if present
        label_mask = labels == label
        print(
            f"Label {label.item()}: {label_mask.sum().item()} samples ({(label_mask.sum() / len(labels) * 100).item():.2f}%)"
        )
        for group in torch.unique(fdm.sens):
            group_mask = fdm.sens == group
            group_label_mask = label_mask & group_mask & calib_mask
            print(
                f"\tGroup {group.item()}: {group_mask.sum().item()} samples ({(group_mask.sum() / len(labels) * 100).item():.2f}%)"
            )
            # print(
            #     f"\t\tGroup {group.item()} - Label {label.item()}: {group_label_mask.sum().item()} samples ({(group_label_mask.sum() / len(labels) * 100).item():.2f}%)"
            # )


if __name__ == "__main__":
    # python run_fed_cf.py --config_path=fair_fedcf_fitzpatrick_config.yaml
    main()
