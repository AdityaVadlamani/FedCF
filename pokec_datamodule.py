# Implementation of 2-client federated datamodule using pokec-z and pokec-n
from collections import Counter
from copy import deepcopy
from enum import Enum
from math import isclose
from typing import List, Optional

import torch
from dgl import DGLGraph, batch, node_subgraph
from sklearn.model_selection import train_test_split

from conformal_fairness.config import SharedBaseConfig
from conformal_fairness.constants import (
    LABEL_FIELD,
    PARTITION_FIELD,
    POKEC,
    POKEC_N,
    POKEC_Z,
    PROBS_KEY,
    Stage,
)
from conformal_fairness.data import GraphDataModule, GraphDataset
from conformal_fairness.utils.data_utils import get_custom_dataset


class PokecPartition(Enum):
    N_PARTITION = 0
    Z_PARTITION = 1


class PokecDataset(GraphDataset):
    def __init__(self, name: str, graph: DGLGraph, args: SharedBaseConfig):
        super(PokecDataset, self).__init__(name, graph, args)

    @property
    def partition(self):
        return self.graph.ndata[PARTITION_FIELD]

    def _setup_masks(self, n_points, extra_calib_test_seed: Optional[int] = None):
        train_mask_n, val_mask_n, calib_mask_n, test_mask_n = (
            self.__masks_for_partition(
                n_points=n_points,
                part=PokecPartition.N_PARTITION.value,
                extra_calib_test_seed=extra_calib_test_seed,
            )
        )

        train_mask_z, val_mask_z, calib_mask_z, test_mask_z = (
            self.__masks_for_partition(
                n_points=n_points,
                part=PokecPartition.Z_PARTITION.value,
                extra_calib_test_seed=extra_calib_test_seed,
            )
        )

        return (
            (train_mask_n | train_mask_z),
            (val_mask_n | val_mask_z),
            (calib_mask_n | calib_mask_z),
            (test_mask_n | test_mask_z),
        )

    def _force_unique_pairs(self, group_label_pairs, seed, n_target, base_ids=None):
        """
        Ensures at least one sample from each unique (group, label) pair
        is selected into the target set. Remaining samples are stratified.
        """
        pair_counts = Counter(group_label_pairs)
        unique_pairs = set(pair_counts.keys())

        target_ids, remaining_ids = [], []

        for i, pair in enumerate(group_label_pairs):
            # If pair is not seen and doesn't have exactly 2 elements then add it to target_ids
            # If it is exactly 2, we let the train_test_split handle this
            if pair in unique_pairs and pair_counts[pair] != 2:
                target_ids.append(i)
                unique_pairs.remove(pair)
            else:
                remaining_ids.append(i)

        # Adjust size after forcing uniques
        n_target_adjusted = n_target - len(target_ids)
        remaining_pairs = [group_label_pairs[i] for i in remaining_ids]

        if n_target_adjusted > 0:
            extra_target, rem_ids = train_test_split(
                remaining_ids,
                train_size=n_target_adjusted,
                stratify=remaining_pairs,
                random_state=seed,
            )
            target_ids.extend(extra_target)
        else:
            rem_ids = remaining_ids

        if base_ids is not None:  # for test+calib split case
            target_ids = [base_ids[i] for i in target_ids]
            rem_ids = [base_ids[i] for i in rem_ids]

        return torch.as_tensor(target_ids), torch.as_tensor(rem_ids)

    def __masks_for_partition(
        self, n_points: int, part: int, extra_calib_test_seed: Optional[int] = None
    ):
        train_mask = torch.zeros(n_points, dtype=torch.bool)
        val_mask = torch.zeros(n_points, dtype=torch.bool)
        calib_mask = torch.zeros(n_points, dtype=torch.bool)
        test_mask = torch.zeros(n_points, dtype=torch.bool)

        assert self.split_config is not None, "Split config must be provided"

        labeled_points = (self.y >= 0) & (self.partition == part)
        # -1 points are unlabeled, we also consider points from the other partition as unlabeled
        n_labeled_points = int(sum(labeled_points))

        all_idx = labeled_points.nonzero(as_tuple=True)[0]

        n_train = int(n_labeled_points * self.split_config.train)
        n_val = int(n_labeled_points * self.split_config.valid)
        n_calib = int(n_labeled_points * self.split_config.calib)

        total_ratio = (
            self.split_config.train + self.split_config.valid + self.split_config.calib
        )

        if total_ratio == 0:
            test_mask[labeled_points] = True  # Dataset with just test points
            return train_mask, val_mask, calib_mask, test_mask

        # Filter out the groups and labels
        groups = self.sens[labeled_points]
        labels = self.y[labeled_points]

        group_label_pairs = list(zip(groups.tolist(), labels.tolist()))

        if isclose(total_ratio, 1.0):
            calib_ids, rem_ids = self._force_unique_pairs(
                group_label_pairs, self.seed, n_calib
            )

            train_ids, val_ids = train_test_split(
                rem_ids,
                train_size=n_train,
                stratify=labels[
                    rem_ids
                ],  # Only stratify by label since group info isn't used in training
                random_state=self.seed,
            )

            train_mask[all_idx[train_ids]] = True
            val_mask[all_idx[val_ids]] = True
            calib_mask[all_idx[calib_ids]] = True
            test_mask[:] = False  # No test points
        else:
            n_test = n_labeled_points - n_train - n_val - n_calib

            # First: split calib+test
            calib_test_ids, rem_ids = self._force_unique_pairs(
                group_label_pairs, self.seed, n_calib + n_test
            )
            calib_test_pairs = [group_label_pairs[i] for i in calib_test_ids]

            # Second: split calib vs test inside that pool
            calib_ids, test_ids = self._force_unique_pairs(
                calib_test_pairs,
                extra_calib_test_seed or self.seed,
                n_calib,
                base_ids=calib_test_ids,
            )

            # Now split train/val
            train_ids, val_ids = train_test_split(
                rem_ids,
                train_size=n_train,
                stratify=labels[
                    rem_ids
                ],  # Only stratify by label and state since group info isn't used in training
                random_state=self.seed,
            )

            train_mask[all_idx[train_ids]] = True
            val_mask[all_idx[val_ids]] = True
            calib_mask[all_idx[calib_ids]] = True
            test_mask[all_idx[test_ids]] = True

        return train_mask, val_mask, calib_mask, test_mask

    def __calib_tune_qscore_for_partition(self, n_points, part, mask_dict, tune_frac):
        assert Stage.CALIBRATION.mask_dstr in mask_dict
        # Modified the top line to account for the partition
        calib_mask = (mask_dict[Stage.CALIBRATION.mask_dstr]) & (self.partition == part)
        calib_points = calib_mask.nonzero(as_tuple=True)[0]
        N = len(calib_points)

        tune_calib_points = torch.zeros(n_points, dtype=torch.bool)
        qscore_calib_points = torch.zeros(n_points, dtype=torch.bool)

        if tune_frac > 0:
            groups = self.sens[calib_points]
            labels = self.y[calib_points]
            group_label_pairs = list(zip(groups, labels))

            tune_calib_ids, qscore_calib_ids, _, _ = train_test_split(
                calib_points,
                group_label_pairs,
                train_size=tune_frac,
                stratify=group_label_pairs,
                random_state=self.seed,
            )
        else:
            # I dont think this case is really needed
            # instead if tune_frac is 0 there are no calib points
            # TODO correct this (based on above comment)
            tune_ct = int(tune_frac * N)
            tune_calib_ids = calib_points[:tune_ct]
            qscore_calib_ids = calib_points[tune_ct:]

        tune_calib_points[tune_calib_ids] = True
        qscore_calib_points[qscore_calib_ids] = True

        return tune_calib_points, qscore_calib_points

    def _setup_calib_tune_qscore(self, n_points, mask_dict, tune_frac):
        tune_calib_points_n, qscore_calib_points_n = (
            self.__calib_tune_qscore_for_partition(
                n_points, PokecPartition.N_PARTITION.value, mask_dict, tune_frac
            )
        )
        tune_calib_points_z, qscore_calib_points_z = (
            self.__calib_tune_qscore_for_partition(
                n_points, PokecPartition.Z_PARTITION.value, mask_dict, tune_frac
            )
        )

        return (tune_calib_points_n & tune_calib_points_z), (
            qscore_calib_points_n,
            qscore_calib_points_z,
        )


class PokecDataModule(GraphDataModule):
    def __init__(self, config: SharedBaseConfig):
        super(PokecDataModule, self).__init__(config)
        self.name = POKEC

    def _create_dataset(
        self,
        name: str = POKEC,
        dataset_dir: str = "./datasets",
        *,
        pred_attrs: Optional[List[str]] = None,
        discard_attrs: Optional[List[str]] = None,
        sens_attrs: Optional[List[str]] = None,
        dataset_args=None,
        force_reprep=False,
    ):
        del name  # Unused

        if pred_attrs is None:
            pred_attrs = []
        if discard_attrs is None:
            discard_attrs = []
        if sens_attrs is None:
            sens_attrs = []

        pokec_n_graph: DGLGraph = get_custom_dataset(
            ds_name=POKEC_N,
            ds_dir=dataset_dir,
            pred_attrs=pred_attrs,
            discard_attrs=discard_attrs,
            sens_attrs=sens_attrs,
            force_reprep=force_reprep,
            dataset_args=dataset_args,
        )[0]

        pokec_z_graph: DGLGraph = get_custom_dataset(
            ds_name=POKEC_Z,
            ds_dir=dataset_dir,
            pred_attrs=pred_attrs,
            discard_attrs=discard_attrs,
            sens_attrs=sens_attrs,
            force_reprep=force_reprep,
            dataset_args=dataset_args,
        )[0]

        # Create a mask for the partitions
        pokec_n_graph_num_nodes = pokec_n_graph.num_nodes()
        pokec_z_graph_num_nodes = pokec_z_graph.num_nodes()
        pokec_graph_num_nodes = pokec_n_graph_num_nodes + pokec_z_graph_num_nodes
        partition_mask = torch.ones(pokec_graph_num_nodes, dtype=torch.int)
        partition_mask[:pokec_n_graph_num_nodes] = PokecPartition.N_PARTITION.value
        partition_mask[pokec_n_graph_num_nodes:] = PokecPartition.Z_PARTITION.value

        combined_graph = batch([pokec_n_graph, pokec_z_graph])
        combined_graph.ndata[PARTITION_FIELD] = partition_mask
        return combined_graph

    def setup(self, args: SharedBaseConfig) -> None:
        if not self.has_setup:
            pokec_graph: DGLGraph = self._create_dataset(
                dataset_dir=self.dataset_dir,
                pred_attrs=self.config.dataset.pred_attrs,
                discard_attrs=self.config.dataset.discard_attrs,
                sens_attrs=self.config.dataset.sens_attrs,
                dataset_args=self.config.dataset,
            )

            dataset = PokecDataset(POKEC, pokec_graph, args=args)
            self._init_with_dataset(dataset)

    def _replace_dataset(self, dataset: GraphDataset):
        # Similar to init with dataset but does not recompute masks
        self._base_dataset = dataset
        self.graph = dataset[0]
        # init all available splits
        self.split_dict = {
            stage: dataset.get_mask_inds(stage.mask_dstr)
            for stage in Stage
            if stage.mask_dstr in self.graph.ndata
        }

        self.has_setup = True

    def load_partition(self, part_id: int, probs=None):
        # Returns only the train, val, cal components
        # the test data is held centrally

        # Assuming part_id = 0 or =1
        assert (part_id == 0) or (
            part_id == 1
        ), f"Partition ID must be 0 or 1, not {part_id}"

        part_dm: PokecDataModule = deepcopy(self)
        part_ds: PokecDataset = deepcopy(self._base_dataset)

        if probs is not None:
            part_ds.graph.ndata[PROBS_KEY] = probs

        part_ds.graph = node_subgraph(
            part_ds.graph, self._base_dataset.partition == part_id
        )
        # Removing Labels of Test data points (to fully ensure separation)
        part_ds.graph.ndata[LABEL_FIELD][
            part_ds.graph.ndata[Stage.TEST.mask_dstr]
        ] = -2  # -2 since -1 is used for unlabeled data
        # Ensuring there are no test nodes
        part_ds.graph.ndata[Stage.TEST.mask_dstr] = torch.zeros_like(
            part_ds.graph.ndata[Stage.TEST.mask_dstr], dtype=torch.bool
        )

        if probs is not None:
            # mask = (
            #     part_ds.graph.ndata[Stage.TRAIN.mask_dstr]
            #     | part_ds.graph.ndata[Stage.VALIDATION.mask_dstr]
            #     | part_ds.graph.ndata[Stage.CALIBRATION.mask_dstr]
            # )
            part_probs = part_ds.graph.ndata.pop(PROBS_KEY)
            # part_probs[~mask] = (
            #     0  # Zero Out test probs (since test data is held centrally) - ensures leakage cannot occur
            # )

        part_dm._replace_dataset(part_ds)
        if probs is not None:
            return part_dm, part_probs

        return part_dm

    def load_test_dm_probs(self, probs):
        test_dm: PokecDataModule = deepcopy(self)
        test_ds: PokecDataset = deepcopy(self._base_dataset)

        # Non-test points should be -2 (to ensure no leakage) since -1 is used for unlabeled data
        test_ds.graph.ndata[LABEL_FIELD][
            ~test_ds.graph.ndata[Stage.TEST.mask_dstr]
        ] = -2

        for stage in Stage:
            if stage != Stage.TEST and stage.mask_dstr in test_ds.graph.ndata:
                test_ds.graph.ndata[stage.mask_dstr] = torch.zeros_like(
                    test_ds.graph.ndata[stage.mask_dstr], dtype=torch.bool
                )

        test_dm._replace_dataset(test_ds)

        return test_dm, probs[self.split_dict[Stage.TEST]]
