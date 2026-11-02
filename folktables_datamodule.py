from collections import Counter
from math import isclose
import os
from sklearn.model_selection import train_test_split
import torch
from typing import Optional, Tuple, Union

from folktables import ACSDataSource, ACSIncome, ACSTravelTime
from folktables.load_acs import _STATE_CODES
from torch.utils.data import DataLoader, Subset

# Import your abstract base classes and any constants you need

from conformal_fairness.constants import Stage, ACS_INCOME, ACS_EDUC
from conformal_fairness.config import SharedBaseConfig
from conformal_fairness.data.tabular_datamodule import TabularDataModule, TabularDataset
from conformal_fairness.utils.data_utils import (
    schl_filter,
    tax_breakdowns,
    schl_transform,
)

from fed_config import FedBaseExptConfig, FedConfFairExptConfig

from fed_constants import (
    FOLKTABLES_ALL,
    FOLKTABLES_CONTINENTAL_ALL,
    FOLKTABLES_OPTIONS,
    FOLKTABLES_SPLIT,
    PARTITION_ID_TO_KEY,
)


class FolktablesDataset(TabularDataset):
    """
    Concrete dataset class for folktables datasets (https://github.com/socialfoundations/folktables)
    """

    def __init__(
        self,
        name: str,
        args: Union[FedBaseExptConfig, FedConfFairExptConfig],
        partition_type: str,
        partition_id: Optional[int] = None,
        global_masks: Optional[dict] = None,
        global_client_mapping: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            name (str): Dataset name.
            args (SharedBaseConfig): Shared config (contains .seed, .dataset_split_fractions, etc.).
            transform: Optional transform to apply to each image.
        """
        data_source = ACSDataSource(
            survey_year="2018",
            horizon="1-Year",
            survey="person",
            root_dir=os.path.join(
                args.dataset_dir, "Folktables"
            ),  # raw files are the same for all folktables datasets, so we will store in a common directory
        )

        if partition_id is None:
            df = data_source.get_data(download=True)
        else:
            if partition_type in FOLKTABLES_OPTIONS:
                df = data_source.get_data(
                    states=FOLKTABLES_SPLIT[partition_type][
                        PARTITION_ID_TO_KEY[partition_type][partition_id]
                    ],
                    download=True,
                )
            else:
                raise ValueError(
                    f"Invalid partition_type provided to FolktablesDataset: {partition_type}"
                )

        code_to_state = {v: k for k, v in _STATE_CODES.items()}

        # If using one of the two groupings, then only focus on continental US for now
        if "continental" in partition_type:
            df = df[~df["ST"].isin([2, 15, 72])]  # AK, HI, PR

            # Remapping races to be:
            # (1): White only
            # (2): Black or African American only
            # (3): American Indian only
            # (4): Asian only
            # (5): Some other race alone
            # (6): Two or More Races
            df["RAC1P"] = df["RAC1P"].map(
                {1: 1, 2: 2, 3: 3, 6: 4, 4: 5, 7: 5, 8: 5, 5: 6, 9: 6}
            )

        if name == ACS_INCOME:
            ACSIncome._target_transform = tax_breakdowns
            if "ST" not in ACSIncome.features:
                ACSIncome._features = ACSIncome.features + ["ST"]
            X, y, sens = ACSIncome.df_to_pandas(df)
        elif name == ACS_EDUC:
            if "SCHL" in ACSTravelTime._features:
                target_index = ACSTravelTime._features.index("SCHL")
                ACSTravelTime._features[target_index] = ACSTravelTime._target
                ACSTravelTime._target = "SCHL"
            ACSTravelTime._target_transform = schl_transform
            ACSTravelTime._preprocess = schl_filter

            if "ST" not in ACSTravelTime.features:
                ACSTravelTime._features = ACSTravelTime.features + ["ST"]
            X, y, sens = ACSTravelTime.df_to_pandas(df)
        else:
            raise AssertionError(f"{name} is not a valid Folktables dataset")

        if partition_id is not None:
            self.client_mapping = [partition_id] * len(X)
        else:
            self.client_mapping = [-1] * len(X)
            state_to_partition_key = {}
            for key, states in FOLKTABLES_SPLIT[partition_type].items():
                for st in states:
                    state_to_partition_key[st] = key

            for i, code in enumerate(X["ST"].values):
                st = code_to_state[f"{int(code):02}"]
                if st is not None:
                    key = state_to_partition_key[st]
                    if key is not None:
                        self.client_mapping[i] = PARTITION_ID_TO_KEY[
                            partition_type
                        ].index(key)

        self.states = X["ST"].to_numpy(
            dtype=int
        )  # to be used for split stratification later
        X = X.drop(columns=["ST"])

        super(FolktablesDataset, self).__init__(
            name=name,
            X=torch.from_numpy(X.values),
            y=torch.from_numpy(y.values).reshape((-1,)),
            sens=torch.from_numpy(sens.values).reshape((-1,)) - 1,
            args=args,
        )

        self.client_mapping = torch.tensor(self.client_mapping)
        self.seed = args.seed
        self.split_config = args.dataset_split_fractions

        if (
            global_masks is not None
            and global_client_mapping is not None
            and partition_id is not None
        ):
            # Get the appropriate train/val/cal for each client from the global data structures
            client_indices = (global_client_mapping == partition_id).nonzero(
                as_tuple=True
            )[0]
            self.masks = {
                split: mask[client_indices] for split, mask in global_masks.items()
            }
        else:
            # Generate global masks
            train_mask, val_mask, calib_mask, test_mask = self._setup_masks(
                X.shape[0],
                args.conformal_seed if hasattr(args, "conformal_seed") else None,
            )
            self.masks = {
                Stage.TRAIN.mask_dstr: train_mask,
                Stage.VALIDATION.mask_dstr: val_mask,
                Stage.CALIBRATION.mask_dstr: calib_mask,
                Stage.TEST.mask_dstr: test_mask,
            }

        # binary_sens: merge race groups into White (0) vs non-White (1) for the
        # fairness computations. Done after the masks so the train/calib/test
        # split (stratified on the original groups) is unchanged.
        if args.dataset.binary_sens:
            self.sens = (torch.as_tensor(self.sens) != 0).long()

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

    def _setup_masks(
        self, n_points: int, extra_calib_test_seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        train_mask = torch.zeros(n_points, dtype=torch.bool)
        val_mask = torch.zeros(n_points, dtype=torch.bool)
        calib_mask = torch.zeros(n_points, dtype=torch.bool)
        test_mask = torch.zeros(n_points, dtype=torch.bool)

        assert self.split_config is not None, "Split config must be provided"

        n_train = int(n_points * self.split_config.train)
        n_val = int(n_points * self.split_config.valid)
        n_calib = int(n_points * self.split_config.calib)

        total_ratio = (
            self.split_config.train + self.split_config.valid + self.split_config.calib
        )

        labeled_points = self.y >= 0
        all_idx = labeled_points.nonzero(
            as_tuple=False
        ).squeeze()  # absolute indices in full graph

        if total_ratio == 0:
            test_mask[labeled_points] = True  # Dataset with just test points
            return train_mask, val_mask, calib_mask, test_mask

        groups, labels, states = (
            self.sens[labeled_points],
            self.y[labeled_points],
            self.states[labeled_points],
        )
        group_label_state_pairs = list(
            zip(groups.tolist(), labels.tolist(), states.tolist())
        )

        if isclose(total_ratio, 1.0):  # No test set
            calib_ids, rem_ids = self._force_unique_pairs(
                group_label_state_pairs, self.seed, n_calib
            )

            val_ids, train_ids = train_test_split(
                rem_ids,
                train_size=n_val,
                stratify=[
                    labels[i] for i in rem_ids
                ],  # Only stratify by label since group info isn't used in training
                random_state=self.seed,
            )

            train_mask[all_idx[train_ids]] = True
            val_mask[all_idx[val_ids]] = True
            calib_mask[all_idx[calib_ids]] = True
            test_mask[:] = False
        else:  # With test set
            n_test = n_points - n_train - n_val - n_calib

            # First: split calib+test
            calib_test_ids, rem_ids = self._force_unique_pairs(
                group_label_state_pairs, self.seed, n_calib + n_test
            )
            calib_test_pairs = [group_label_state_pairs[i] for i in calib_test_ids]

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
                stratify=[
                    (labels[i], states[i]) for i in rem_ids
                ],  # Only stratify by label and state since group info isn't used in training
                random_state=self.seed,
            )

            train_mask[all_idx[train_ids]] = True
            val_mask[all_idx[val_ids]] = True
            calib_mask[all_idx[calib_ids]] = True
            test_mask[all_idx[test_ids]] = True

        return train_mask, val_mask, calib_mask, test_mask


class FolktablesDataModule(TabularDataModule):
    """
    Concrete DataModule for Folktables dataset.
    """

    def __init__(
        self,
        config: SharedBaseConfig,
        /,
        *,
        partition_type: str,
        partition_id: Optional[int] = None,
    ):
        super().__init__(config)

        if partition_type not in FOLKTABLES_OPTIONS:
            raise ValueError(
                f"Folktables only supports partition_type in {FOLKTABLES_OPTIONS}"
            )

        if "small" in partition_type:
            num_partitions = 4
        elif "large" in partition_type:
            num_partitions = 8
        elif partition_type == FOLKTABLES_CONTINENTAL_ALL:
            num_partitions = 48
        elif partition_type == FOLKTABLES_ALL:
            num_partitions = 51
        else:
            raise ValueError("Invalid partition_type provided")

        if partition_id is not None and (
            partition_id < 0 or partition_id >= num_partitions
        ):
            raise ValueError("Invalid partition_id provided")

        self.partition_id = partition_id
        self.partition_type = partition_type

    @property
    def client_mapping(self):
        return torch.as_tensor(self._base_dataset.client_mapping)

    @property
    def masks(self):
        return self._base_dataset.masks

    @property
    def num_classes(self) -> int:
        if self.name == ACS_INCOME:
            return 4
        elif self.name == ACS_EDUC:
            return 6
        else:
            raise ValueError("Invalid folktables dataset name")

    @property
    def num_sensitive_groups(self):
        if self.config.dataset.binary_sens:
            return 2
        if "continental" in self.partition_type:
            return 6
        return 9

    # ---------------------
    # BaseDataModule core methods
    # ---------------------
    def _create_dataset(
        self,
        name: str,
        partition_type: str,
        partition_id: Optional[int],
        dataset_dir: str = "",
        *,
        pred_attrs=[],
        discard_attrs=[],
        sens_attrs=[],
        dataset_args=None,
        force_reprep=False,
        global_masks=None,
        global_client_mapping=None,
    ):
        """
        This method is responsible for creating and returning X, y, sens
        OR returning the actual dataset instance.
        However, in the tabular code, it returns raw (X, y, sens).
        """

        dataset = FolktablesDataset(
            name=name,
            args=self.config,
            partition_type=partition_type,
            partition_id=partition_id,
            global_masks=global_masks,
            global_client_mapping=global_client_mapping,
        )
        return dataset

    def prepare_data(self) -> None:
        """
        Called once before setup(). You could download data or do big steps here.
        For local data, you might do nothing.
        """
        if not os.path.exists(
            os.path.join(self.dataset_dir, self.name, "2018", "1-Year")
        ):
            self._create_dataset(
                self.name,
                self.partition_type,
                None,
                self.dataset_dir,
                pred_attrs=self.config.dataset.pred_attrs,
                discard_attrs=self.config.dataset.discard_attrs,
                sens_attrs=self.config.dataset.sens_attrs,
                force_reprep=self.config.dataset.force_reprep,
                dataset_args=self.config.dataset,
            )

    def setup(self, args: SharedBaseConfig = None) -> None:
        """
        Called before the data loaders are created.
        """
        if args is None:
            args = self.config
        if not self.has_setup:
            dataset = self._create_dataset(
                self.name,
                self.partition_type,
                None,
                self.dataset_dir,
                pred_attrs=args.dataset.pred_attrs,
                discard_attrs=args.dataset.discard_attrs,
                sens_attrs=args.dataset.sens_attrs,
                dataset_args=args.dataset,
            )
            self._init_with_dataset(dataset)

    # ---------------------
    # DataLoader methods
    # ---------------------
    def train_dataloader(self):
        train_inds = self.split_dict.get(Stage.TRAIN, None)
        if train_inds is None:
            # fallback to all data?
            train_inds = torch.arange(self.num_points)

        train_subset = Subset(self._base_dataset, train_inds)
        return DataLoader(
            train_subset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def val_dataloader(self):
        val_inds = self.split_dict.get(Stage.VALIDATION, None)
        if val_inds is None:
            val_inds = torch.arange(self.num_points)

        val_subset = Subset(self._base_dataset, val_inds)
        return DataLoader(
            val_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def test_dataloader(self):
        test_inds = self.split_dict.get(Stage.TEST, None)
        if test_inds is None:
            test_inds = torch.arange(self.num_points)

        test_subset = Subset(self._base_dataset, test_inds)
        return DataLoader(
            test_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def all_dataloader(self):
        # Return everything (or all labeled)
        inds = torch.arange(self.num_points)
        all_subset = Subset(self._base_dataset, inds)
        return DataLoader(
            all_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )

    def custom_dataloader(
        self,
        points,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
        **kwargs,
    ):
        """
        A method to create a DataLoader from an arbitrary subset of indices.
        """
        if batch_size is None:
            batch_size = self.batch_size

        subset = Subset(self._base_dataset, points)
        return DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=self.num_workers,
            **kwargs,
        )
