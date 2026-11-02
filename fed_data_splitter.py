from typing import Dict
from copy import deepcopy

import torch
import numpy as np
from datasets import Dataset
from flwr_datasets.partitioner import Partitioner

from conformal_fairness.data.base_datamodule import BaseDataModule
from conformal_fairness.constants import (
    FEATURE_FIELD,
    SENS_FIELD,
    LABEL_FIELD,
    Stage,
)


class FedDataSplitter:
    def __init__(
        self,
        datamodule: BaseDataModule,
        data_partitioner: Partitioner,
        num_partitions,
        split_dict: Dict[Stage, torch.Tensor],
        probs=None,
    ):
        split_dict_without_test = {
            key: split_dict[key] for key in split_dict if key != Stage.TEST
        }

        self.data_part_dict = {
            key: deepcopy(data_partitioner) for key in split_dict_without_test
        }

        self.dm = datamodule

        dataset = datamodule._base_dataset
        ds_dict_by_split = {
            key: {
                FEATURE_FIELD: split_dict_without_test[key],
                LABEL_FIELD: dataset.y[split_dict_without_test[key]],
                SENS_FIELD: dataset.sens[split_dict_without_test[key]],
            }
            for key in split_dict_without_test
        }

        self.probs = probs
        for key in split_dict_without_test:
            self.data_part_dict[key].dataset = Dataset.from_dict(ds_dict_by_split[key])

        self.feat_type = type(dataset.X)
        self.num_partitions = num_partitions
        self.split_dict_without_test = split_dict_without_test

    def load_partition(self, part_id: int):
        partition_dataset_by_split = {}
        for key in self.split_dict_without_test:
            partition_dataset_by_split[key] = self.data_part_dict[key].load_partition(
                part_id
            )
        dm = deepcopy(self.dm)
        ds = dm._base_dataset

        index_list = torch.cat(
            [
                torch.tensor(partition_dataset_by_split[key][FEATURE_FIELD])
                for key in self.split_dict_without_test
            ]
        )

        if self.feat_type is np.ndarray:
            ds.X = ds.X[index_list]
        elif self.feat_type is torch.Tensor:
            ds.X = ds.X[index_list]
        elif self.feat_type is list:
            ds.X = [ds.X[i] for i in index_list]
        else:
            raise NotImplementedError

        ds.y = ds.y[index_list]
        ds.sens = ds.sens[index_list]

        for stage in Stage:
            if stage.mask_dstr in ds.masks:
                ds.masks[stage.mask_dstr] = ds.masks[stage.mask_dstr][index_list]

        num_points = len(index_list)
        ds.split_config.train = sum(ds.masks[Stage.TRAIN.mask_dstr]) / num_points
        ds.split_config.valid = sum(ds.masks[Stage.VALIDATION.mask_dstr]) / num_points
        ds.split_config.calib = sum(ds.masks[Stage.CALIBRATION.mask_dstr]) / num_points

        dm._base_dataset = ds

        # Initialize the dictionary that tracks splits
        # (train, val, test, etc.). We'll build from dataset.masks
        dm.split_dict = {
            stage: ds.get_mask_inds(stage.mask_dstr)
            for stage in Stage
            if stage.mask_dstr in ds.masks
        }

        dm.has_setup = True

        if self.probs is not None:
            return dm, self.probs[index_list]

        return dm
