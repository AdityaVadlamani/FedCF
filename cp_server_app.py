"""base-fed-model: A Flower / PyTorch app."""

import torch

from flwr.common import Context
from flwr.server import ServerAppComponents, ServerConfig

from fed_config import FedConfFairExptConfig
from conformal_fairness.data.base_datamodule import BaseDataModule
from flwr_cp_strategy import FedConfFairStrategy


def server_fn(
    context: Context,
    args: FedConfFairExptConfig,
    test_dm: BaseDataModule,
    test_probs: torch.Tensor,
    all_probs: torch.Tensor = None,
):
    # Read from config
    num_rounds = args.cf_opt.num_opt_rounds

    # Assuming all but 0-class are positive labels:
    positive_labels = torch.arange(1, test_dm.num_classes)

    # Define Strategy
    strategy = FedConfFairStrategy(
        config=args,
        test_dm=test_dm,
        test_probs=test_probs,
        positive_labels=positive_labels,
        all_probs=all_probs,
    )

    # +2 comes from one init_stage and one fed_cp stage before the fed cf framework runs
    config = ServerConfig(num_rounds=num_rounds + 2)

    return ServerAppComponents(strategy=strategy, config=config)
