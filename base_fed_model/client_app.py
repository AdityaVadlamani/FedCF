"""base-fed-model: A Flower / PyTorch app."""

from typing import Union
import dgl
from lightning import seed_everything
import torch
import torch.nn as nn
import xgboost as xgb

from fairlex_datamodule import FairlexDatamodule
from conformal_fairness.config import BaseXGBoostConfig
from conformal_fairness.constants import DEFAULT_DEVICE, POKEC, Stage
from conformal_fairness.data.base_datamodule import BaseDataModule
from flwr.client import Client
from flwr.common import (
    Context,
    Parameters,
    FitIns,
    FitRes,
    EvaluateIns,
    EvaluateRes,
    Status,
    Code,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from fed_config import FedBaseExptConfig
from folktables_datamodule import FolktablesDataModule
from pokec_datamodule import PokecDataModule
from fitzpatrick_datamodule import FitzpatrickDataModule
from task import (
    get_gnn_model,
    get_fairlex_model,
    get_weights,
    load_data,
    load_acs_data,
    set_weights,
    test,
    test_gnn,
    train,
    get_model,
    train_gnn,
)


class FedBaseClient(Client):
    def __init__(
        self, name, datamodule: BaseDataModule, partition_id: int, num_partitions: int
    ):
        self.name = name
        self.datamodule = datamodule
        self.local_epochs = datamodule.config.epochs
        self.partition_id = partition_id
        self.num_partitions = num_partitions
        self.device = torch.device(DEFAULT_DEVICE)

    def from_nn(self, model: nn.Module, lr: float):
        self.net = (model,)  # Hacky fix
        self.lr = lr

    def from_xgb(self, params: BaseXGBoostConfig, num_local_rounds: int = 10):
        self.num_local_round = num_local_rounds

        self.params = {
            "tree_method": params.tree_method,
            "objective": "multi:softprob",
            "num_class": self.datamodule.num_classes,
            "eval_metric": "merror",
            "device": "cpu",
            "max_depth": params.max_depth,
            "max_leaves": params.max_leaves,
            "grow_policy": params.grow_policy,
            "booster": params.booster,
            "gamma": params.gamma,
            "colsample_bytree": params.colsample_bytree,
            "colsample_bylevel": params.colsample_bylevel,
            "colsample_bynode": params.colsample_bynode,
            "subsample": params.subsample,
            "reg_alpha": params.reg_alpha,
            "reg_lambda": params.reg_lambda,
            "nthread": -1,
        }

        train_labels = (
            self.datamodule.y[self.datamodule.split_dict[Stage.TRAIN]]
            .cpu()
            .detach()
            .numpy()
        )
        train_features = (
            self.datamodule.X[self.datamodule.split_dict[Stage.TRAIN], :]
            .cpu()
            .detach()
            .numpy()
        )

        self.num_train = len(train_labels)
        self.train_dmatrix = xgb.DMatrix(train_features, train_labels)

        valid_labels = (
            self.datamodule.y[self.datamodule.split_dict[Stage.VALIDATION]]
            .cpu()
            .detach()
            .numpy()
        )
        valid_features = (
            self.datamodule.X[self.datamodule.split_dict[Stage.VALIDATION], :]
            .cpu()
            .detach()
            .numpy()
        )

        self.num_valid = len(valid_labels)
        self.valid_dmatrix = xgb.DMatrix(valid_features, valid_labels)

    @property
    def valloader(
        self,
    ) -> Union[torch.utils.data.DataLoader, dgl.dataloading.DataLoader]:
        return self.datamodule.val_dataloader()

    @property
    def trainloader(
        self,
    ) -> Union[torch.utils.data.DataLoader, dgl.dataloading.DataLoader]:
        return self.datamodule.train_dataloader()

    def _local_boost(self, bst_input):
        # Update trees based on local training data.
        for _ in range(self.num_local_round):
            bst_input.update(self.train_dmatrix, bst_input.num_boosted_rounds())

        # Bagging: extract the last N=num_local_round trees for sever aggregation
        bst = bst_input[
            bst_input.num_boosted_rounds()
            - self.num_local_round : bst_input.num_boosted_rounds()
        ]

        return bst

    def _fit_xgboost(self, ins: FitIns):
        global_round = int(ins.config["global_round"])
        if global_round == 1:
            # First round local training
            bst = xgb.train(
                self.params,
                self.train_dmatrix,
                num_boost_round=self.num_local_round,
                evals=[
                    (self.valid_dmatrix, "validate"),
                    (self.train_dmatrix, "train"),
                ],
            )
        else:
            bst = xgb.Booster(params=self.params)
            global_model = bytearray(ins.parameters.tensors[0])

            # Load global model into booster
            bst.load_model(global_model)

            # Local training
            bst = self._local_boost(bst)

        # Save model
        local_model = bst.save_raw("json")
        local_model_bytes = bytes(local_model)

        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=Parameters(tensor_type="", tensors=[local_model_bytes]),
            num_examples=self.num_train,
            metrics={},
        )

    def _fit_nn(self, ins: FitIns):
        if isinstance(self.net, tuple):
            self.net = self.net[0]

        set_weights(self.net, parameters_to_ndarrays(ins.parameters))

        if self.name == POKEC:
            results = train_gnn(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.lr,
                self.device,
            )
            assert isinstance(
                self.trainloader, dgl.dataloading.DataLoader
            ), "Expected DGL dataloader for Pokec"
            num_examples = len(self.trainloader.indices)
        else:
            results = train(
                self.net,
                self.trainloader,
                self.local_epochs,
                self.lr,
                self.device,
            )
            assert isinstance(
                self.trainloader, torch.utils.data.DataLoader
            ), "Expected torch dataloader for other NN datasets"
            num_examples = len(self.trainloader.dataset)

        metrics = {"results": float(results)}
        return FitRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            parameters=ndarrays_to_parameters(get_weights(self.net)),
            num_examples=num_examples,
            metrics=metrics,
        )

    def fit(self, ins: FitIns) -> FitRes:
        """Train the model with data of this client."""
        if isinstance(self.datamodule, FolktablesDataModule):
            return self._fit_xgboost(ins)

        return self._fit_nn(ins)

    def _evaluate_xgboost(self, ins: EvaluateIns):
        # Load global model
        bst = xgb.Booster(params=self.params)
        para_b = bytearray(ins.parameters.tensors[0])
        bst.load_model(para_b)

        # Run evaluation
        eval_results = bst.eval_set(
            evals=[(self.valid_dmatrix, "valid")],
            iteration=bst.num_boosted_rounds() - 1,
        )
        error = round(float(eval_results.split("\t")[1].split(":")[1]), 4)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=0.0,
            num_examples=self.num_valid,
            metrics={"accuracy": 1 - error},
        )

    def _evaluate_nn(self, ins: EvaluateIns):
        if isinstance(self.net, tuple):
            self.net = self.net[0]
        set_weights(self.net, parameters_to_ndarrays(ins.parameters))
        if self.name == POKEC:
            loss, accuracy = test_gnn(self.net, self.valloader, self.device)
            assert isinstance(
                self.valloader, dgl.dataloading.DataLoader
            ), "Expected DGL dataloader for Pokec"
            num_examples = len(self.valloader.indices)
        else:
            loss, accuracy = test(self.net, self.valloader, self.device)
            assert isinstance(
                self.valloader, torch.utils.data.DataLoader
            ), "Expected torch dataloader for NN datasets"
            num_examples = len(self.valloader.dataset)

        return EvaluateRes(
            status=Status(
                code=Code.OK,
                message="OK",
            ),
            loss=loss,
            num_examples=num_examples,
            metrics={"accuracy": accuracy},
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate the model on the data this client has."""
        if isinstance(self.datamodule, FolktablesDataModule):
            return self._evaluate_xgboost(ins)

        return self._evaluate_nn(ins)


def client_fn(context: Context, args: FedBaseExptConfig, datamodule: BaseDataModule):
    """Construct a Client that will be run in a ClientApp.
    Assume Federated DataModule Is provided by the client
    """
    # Read the node_config to fetch data partition associated to this node
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]

    seed_everything(args.seed + partition_id)

    if isinstance(datamodule, FolktablesDataModule):
        global_masks = datamodule.masks
        global_client_mapping = datamodule.client_mapping

        fed_datamodule = load_acs_data(
            datamodule=datamodule,
            partition_type=args.folktables_partition_type,
            partition_id=partition_id,
            global_masks=global_masks,
            global_client_mapping=global_client_mapping,
        )
    elif isinstance(datamodule, PokecDataModule):
        fed_datamodule = datamodule.load_partition(part_id=partition_id)
    else:
        fed_datamodule = load_data(
            datamodule=datamodule,
            num_partitions=num_partitions,
            partition_id=partition_id,
        )

    client = FedBaseClient(
        datamodule.name, fed_datamodule, partition_id, num_partitions
    )

    if isinstance(datamodule, FolktablesDataModule):
        client.from_xgb(args.base_model_config, args.base_model_config.n_estimators)
    elif isinstance(datamodule, PokecDataModule):
        model = get_gnn_model(
            config=args.base_model_config,
            num_features=datamodule.num_features,
            num_classes=datamodule.num_classes,
        )
        client.from_nn(
            model, args.base_model_config.lr
        )  # might have to get nn.Module directly

    elif isinstance(datamodule, FitzpatrickDataModule):
        resnet_arch = args.base_model_config.architecture
        model = get_model(resnet_arch, datamodule.num_classes)
        client.from_nn(model, args.base_model_config.lr)
    elif isinstance(datamodule, FairlexDatamodule):
        model = get_fairlex_model(args.dataset.name, datamodule.num_classes)
        client.from_nn(model, args.base_model_config.lr)
    else:
        raise NotImplementedError

    return client.to_client()
