from contextlib import ExitStack
import os
from collections import OrderedDict
from copy import deepcopy
from typing import Dict, Optional

import dgl
from flwr_datasets.partitioner import DirichletPartitioner
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification
import xgboost as xgb


from conformal_fairness.models import GNN
from conformal_fairness.constants import *
from conformal_fairness.utils.ml_utils import _get_output_directory
from conformal_fairness.utils.sys_utils import enter_cpu_cxs
from conformal_fairness.data.base_datamodule import BaseDataModule
from conformal_fairness.config import BaseExptConfig, BaseGNNConfig
from fed_data_splitter import FedDataSplitter
from folktables_datamodule import FolktablesDataModule
import resnet

fds = None  # Cache FederatedDataset


def load_acs_data(
    datamodule: FolktablesDataModule,
    partition_type: str,
    partition_id: Optional[int] = None,
    global_masks=None,
    global_client_mapping=None,
    probs: Dict[int, torch.Tensor] = {},
):
    fed_datamodule = deepcopy(datamodule)

    client_dataset = fed_datamodule._create_dataset(
        fed_datamodule.name,
        partition_type,
        partition_id=partition_id,
        dataset_dir=fed_datamodule.dataset_dir,
        global_masks=global_masks,
        global_client_mapping=global_client_mapping,
    )

    fed_datamodule._init_with_dataset(client_dataset)

    if probs and partition_id in probs:
        return fed_datamodule, probs[partition_id]
    else:
        return fed_datamodule


def load_data(
    datamodule: BaseDataModule,
    num_partitions: int,
    partition_id: int,
    probs: torch.Tensor = None,
):
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            min_partition_size=100,
            self_balancing=True,
        )

        fds = FedDataSplitter(
            datamodule,
            data_partitioner=partitioner,
            num_partitions=num_partitions,
            split_dict=datamodule.split_dict,
            probs=probs,
        )

    if probs is not None:
        partition, part_probs = fds.load_partition(partition_id)
    else:
        partition = fds.load_partition(partition_id)

    fed_datamodule = deepcopy(partition)

    if probs is not None:
        return fed_datamodule, part_probs

    return fed_datamodule


def train(net, trainloader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for i, batch in enumerate(trainloader):
            labels = batch["label"].to(device)
            if "input" in batch:
                inputs = batch["input"].to(device)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
            elif "attention_mask" in batch and "input_ids" in batch:
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                outputs = net(**inputs, labels=labels)
                loss = outputs.loss

            optimizer.zero_grad()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            labels = batch["label"].to(device)
            if "input" in batch:
                inputs = batch["input"].to(device)
                outputs = net(inputs)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs, 1)[1] == labels).sum().item()
            elif "attention_mask" in batch and "input_ids" in batch:
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                outputs = net(**inputs, labels=labels)
                loss += outputs.loss.item()
                correct += (torch.max(outputs.logits, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


def test_probs(net: torch.nn.Module, testloader, device):
    """Validate the model on the test set."""
    net.eval()
    net.to(device)
    out_list = []
    lab_list = []
    ids_list = []

    with torch.no_grad():
        for i, batch in enumerate(testloader):

            ids = batch["ids"]
            labels = batch["label"].to(device)

            if "input" in batch:
                inputs = batch["input"].to(device)
                outputs = F.softmax(net(inputs), dim=-1)
            elif "attention_mask" in batch and "input_ids" in batch:
                assert (
                    batch["input_ids"].dim() == 2
                ), f"Expected [batch_size, seq_len], got {batch['input_ids'].shape}"
                assert batch["attention_mask"].shape == batch["input_ids"].shape
                inputs = {
                    "input_ids": batch["input_ids"].to(device),
                    "attention_mask": batch["attention_mask"].to(device),
                }
                outputs = F.softmax(net(**inputs).logits, dim=-1)
            else:
                raise NotImplementedError()

            out_list.append(outputs)
            lab_list.append(labels)
            ids_list.append(ids)

    # Concatenate along the first (sample) dimension
    ids_all = torch.cat(ids_list, dim=0)
    outs_all = torch.cat(out_list, dim=0)
    labels_all = torch.cat(lab_list, dim=0)

    return ids_all.cpu(), outs_all.cpu(), labels_all.cpu()


def train_gnn(net, trainloader: dgl.dataloading.DataLoader, epochs, lr, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for i, batch in enumerate(trainloader):
            _, _, mfgs = batch
            mfgs = [mfg.to(device) for mfg in mfgs]
            inputs = mfgs[0].srcdata[FEATURE_FIELD].to(device)
            labels = mfgs[-1].dstdata[LABEL_FIELD].to(device)
            optimizer.zero_grad()
            loss = criterion(net(mfgs, inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader.indices)
    return avg_trainloss


def test_gnn(net: torch.nn.Module, testloader: dgl.dataloading.DataLoader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0

    net.eval()
    with torch.no_grad():
        for batch in testloader:
            _, _, mfgs = batch
            mfgs = [mfg.to(device) for mfg in mfgs]
            inputs = mfgs[0].srcdata[FEATURE_FIELD].to(device)
            labels = mfgs[-1].dstdata[LABEL_FIELD].to(device)
            outputs = net(mfgs, inputs)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs, 1)[1] == labels).sum().item()

    accuracy = correct / len(testloader.indices)
    loss = loss / len(testloader.indices)
    return loss, accuracy


def test_gnn_probs(net: GNN, testloader, device):
    """Validate the model on the test set."""
    net.eval()
    net.to(device)
    out_list = []
    lab_list = []
    ids_list = []

    with torch.no_grad():
        for batch in testloader:
            _, ids, mfgs = batch
            mfgs = [mfg.to(device) for mfg in mfgs]
            inputs = mfgs[0].srcdata[FEATURE_FIELD].to(device)
            labels = mfgs[-1].dstdata[LABEL_FIELD].to(device)
            outputs = F.softmax(net(mfgs, inputs), dim=-1)
            out_list.append(outputs)
            lab_list.append(labels)
            ids_list.append(ids)

    # Concatenate along the first (sample) dimension
    ids_all = torch.cat(ids_list, dim=0)
    outs_all = torch.cat(out_list, dim=0)
    labels_all = torch.cat(lab_list, dim=0)

    return ids_all.cpu(), outs_all.cpu(), labels_all.cpu()


def get_weights(net):
    weights = [val.cpu().numpy() for _, val in net.state_dict().items()]
    # print("Getting weights: ", weights)
    return weights


def set_weights(net, parameters):
    # print("Setting weights: ", parameters)
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_gnn_model(config: BaseGNNConfig, num_features: int, num_classes: int):
    return GNN(
        config=config,
        num_features=num_features,
        num_classes=num_classes,
    )


def get_fairlex_model(name, num_classes):
    if name in (CAIL,):
        model_name = f"coastalcph/fairlex-{name}-minilm"
    else:
        raise ValueError(f"Invalid name for FairlexDataset: {name}")

    config = AutoConfig.from_pretrained(model_name, num_labels=num_classes)
    model = AutoModelForSequenceClassification.from_config(config)
    return model


def get_model(arch, num_classes):
    model = getattr(resnet, arch)(pretrained=True)

    if hasattr(model, "fc"):
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError("Unknown Last Layer")
    return model


def run_xgb_inference_alldl(model: xgb.Booster, datamodule: FolktablesDataModule):
    num_points = datamodule.num_points

    all_labels = datamodule.y.cpu().detach().numpy()
    all_features = datamodule.X.cpu().detach().numpy()

    print(all_labels)

    all_dmatrix = xgb.DMatrix(all_features, all_labels)
    probs = model.predict(all_dmatrix)

    updated_test_results = {
        NODE_IDS_KEY: torch.arange(num_points),
        LABELS_KEY: datamodule.y,
        PROBS_KEY: torch.tensor(probs),
    }

    updated_test_results[PARTITION_TYPE_FIELD] = datamodule.partition_type
    updated_test_results[PARTITION_FIELD] = datamodule.client_mapping
    return updated_test_results


def run_inference_alldl(model, datamodule: BaseDataModule):
    if datamodule.name == POKEC:
        with ExitStack() as stack:
            dl = enter_cpu_cxs(datamodule, ["all_dataloader"], stack)[0]
        ids, smx, labels = test_gnn_probs(model, dl, DEFAULT_DEVICE)
    else:
        dl = datamodule.all_dataloader()
        ids, smx, labels = test_probs(model, dl, DEFAULT_DEVICE)

    num_points = datamodule.num_points
    num_classes = datamodule.num_classes

    updated_test_results = {
        NODE_IDS_KEY: torch.arange(num_points),
        LABELS_KEY: torch.fill(torch.empty(num_points, dtype=torch.long), -1),
        PROBS_KEY: torch.fill(torch.empty(num_points, num_classes), -1),
    }

    updated_test_results[LABELS_KEY][ids] = labels
    updated_test_results[PROBS_KEY][ids] = smx

    return updated_test_results


def output_results(
    args: BaseExptConfig,
    results: Dict[str, torch.Tensor],
    job_output_dir: Optional[str] = None,
):
    assert NODE_IDS_KEY in results
    # assert that results[NODE_IDS_KEY] is sorted
    assert torch.all(
        results[NODE_IDS_KEY].argsort() == torch.arange(len(results[NODE_IDS_KEY]))
    )
    assert LABELS_KEY in results and PROBS_KEY in results

    if job_output_dir is None:
        job_output_dir = _get_output_directory(
            args.output_dir, args.dataset.name, args.job_id
        )

    os.makedirs(job_output_dir, exist_ok=True)
    torch.save(results, os.path.join(job_output_dir, ALL_OUTPUTS_FILE))
