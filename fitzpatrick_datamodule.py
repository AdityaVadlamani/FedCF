from collections import Counter
from math import isclose
import os
from typing import Optional, Tuple
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import skimage.io as io
import skimage
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from PIL import Image

from conformal_fairness.data.base_datamodule import BaseDataModule, BaseDataset

# Import your abstract base classes and any constants you need

from conformal_fairness.constants import FAIRNESS_DATASETS, Stage
from conformal_fairness.config import SharedBaseConfig


class FitzpatrickDataset(BaseDataset):
    """
    Concrete dataset class for Fitzpatrick images, inheriting from BaseDataset.
    """

    def __init__(
        self,
        name: str,
        csv_file: str,
        root_dir: str,
        args: SharedBaseConfig,
        transform=None,
    ):
        """
        Args:
            name (str): Dataset name.
            csv_file (str): Path to CSV file with image info and labels.
            root_dir (str): Folder containing the images.
            args (SharedBaseConfig): Shared config (contains .seed, .dataset_split_fractions, etc.).
            transform: Optional transform to apply to each image.

        The CSV is assumed to have columns like:
          - 'hasher' or 'filename' (unique image ID or file name),
          - 'mid' (your label, for example),
          - 'fitzpatrick' (the sensitive attribute),
          or any others you might need.
        """
        super().__init__(name=name)
        self.csv_file = csv_file
        self.root_dir = root_dir
        self.transform = transform

        self.df = pd.read_csv(csv_file)

        # BaseDataset expects you to define self._X, self._y, self._sens
        self._X = None
        self._y = None
        self._sens = None

        # For splitting logic
        self.split_config = args.dataset_split_fractions
        self.seed = args.seed

        # Dictionary to hold different split masks (train/val/calib/test, etc.)
        # e.g. self.masks["train"] = a boolean or index tensor
        self.masks = {}

        # Do initial processing
        train_mask, val_mask, calib_mask, test_mask = self.process()
        self.masks[Stage.TRAIN.mask_dstr] = train_mask
        self.masks[Stage.VALIDATION.mask_dstr] = val_mask
        self.masks[Stage.CALIBRATION.mask_dstr] = calib_mask
        self.masks[Stage.TEST.mask_dstr] = test_mask

    # ---------------------
    # Abstract properties
    # ---------------------
    @property
    def X(self) -> np.ndarray:
        # TODO self._X is actually an NDArray
        return self._X

    @property
    def y(self) -> torch.Tensor:
        return self._y

    @property
    def sens(self) -> torch.Tensor:
        return self._sens

    @X.setter
    def X(self, val):
        self._X = val

    @y.setter
    def y(self, val):
        self._y = val

    @sens.setter
    def sens(self, val):
        self._sens = val

    # ---------------------
    # Required methods
    # ---------------------
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

        groups, labels = (
            self.sens[labeled_points],
            self.y[labeled_points],
        )
        group_label_pairs = list(
            zip(groups.tolist(), labels.tolist())
        )

        if isclose(total_ratio, 1.0):  # No test set
            calib_ids, rem_ids = self._force_unique_pairs(
                group_label_pairs, self.seed, n_calib
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

        else:  # With test set
            n_test = n_points - n_train - n_val - n_calib

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
                stratify=[
                    labels[i] for i in rem_ids
                ],  # Only stratify by label since group info isn't used in training
                random_state=self.seed,
            )

            train_mask[all_idx[train_ids]] = True
            val_mask[all_idx[val_ids]] = True
            calib_mask[all_idx[calib_ids]] = True
            test_mask[all_idx[test_ids]] = True

        return train_mask, val_mask, calib_mask, test_mask

    def process(self):
        """
        Called once to read CSV and set up _X, _y, _sens arrays.
        Also calls _setup_masks(...) to create train/val/calib/test masks.
        """

        # Load data in manually (Assume nothing is setup)
        # Otherwise self_X is already given and we don't need to reload data
        if self._X is None:
            # Suppose CSV columns:
            #   'hasher' -> unique image filename
            #   'mid' -> integer label
            #   'fitzpatrick_scale' -> integer for Fitzpatrick skin type

            # Store the image file paths in self._X
            file_paths = self.df["hasher"].apply(
                lambda x: os.path.join(self.root_dir, x) + ".jpg"
            )
            self._X = np.array(file_paths)  # keep as numpy array of strings

            # Label
            if "mid" in self.df.columns:
                self._y = torch.tensor(self.df["mid"].values, dtype=torch.long)
            else:
                self.df["mid"] = (
                    self.df["nine_partition_label"].astype("category").cat.codes
                )
                self._y = torch.tensor(self.df["mid"].values, dtype=torch.long)

            remap = torch.tensor([1, 2, 3, 4, 0, 5, 6, 7, 8], dtype=torch.long, device=self._y.device)
            self._y = remap[self._y]
            
            # Sensitive attribute
            if "fitzpatrick_scale" in self.df.columns:
                self._sens = (
                    torch.tensor(self.df["fitzpatrick_scale"].values, dtype=torch.long)
                    - 1
                )

            else:
                # Or default to -1 if not found
                self._sens = torch.full((len(self.df),), -1, dtype=torch.long)

            valid_sens = self._sens >= 0
            self._X = self._X[valid_sens]
            self._y = self._y[valid_sens]
            self._sens = self._sens[valid_sens]

        # Create train/val/calib/test masks if you need them.
        # This uses a helper from the BaseDataset class:
        train_mask, val_mask, calib_mask, test_mask = self._setup_masks(len(self.X))
        return train_mask, val_mask, calib_mask, test_mask

    def resplit_calib_test(self, seed: int):
        """
        If you want to re-split calibration/test subsets with a new seed.
        """
        # Just call _setup_masks again with extra_calib_test_seed=seed
        # Then store them in self.masks
        new_train, new_val, new_calib, new_test = self._setup_masks(
            len(self._X), extra_calib_test_seed=seed
        )
        self.masks[Stage.TRAIN.mask_dstr] = new_train
        self.masks[Stage.VALIDATION.mask_dstr] = new_val
        self.masks[Stage.CALIBRATION.mask_dstr] = new_calib
        self.masks[Stage.TEST.mask_dstr] = new_test
        return self

    def split_calib_tune_qscore(self, tune_frac: float):
        """
        If you want to further split the calibration set into tune vs. qscore sets.
        """
        tune_calib_points, qscore_calib_points = self._setup_calib_tune_qscore(
            n_points=len(self._X), mask_dict=self.masks, tune_frac=tune_frac
        )
        self.masks[Stage.CALIBRATION_TUNE.mask_dstr] = tune_calib_points
        self.masks[Stage.CALIBRATION_QSCORE.mask_dstr] = qscore_calib_points
        return self

    def get_mask_inds(self, mask_key: str):
        """
        Returns the indices for the given subset (train, val, calib, test, etc.).
        """
        if mask_key not in self.masks:
            return None
        mask = self.masks[mask_key]
        # convert bool mask -> indices
        return torch.nonzero(mask, as_tuple=True)[0]

    def update_features(self, new_feats):
        """
        If you'd like to replace X with new feature vectors or embeddings at runtime.
        """
        self.X = new_feats

    def __len__(self) -> int:
        return len(self._X)

    def __getitem__(self, index):
        """
        Return a single sample as (image, label, sens).
        Or a dict with keys 'img', 'label', 'sens'—depending on what your training code expects.
        """
        if torch.is_tensor(index):
            index = index.tolist()

        img_path = self._X[index]
        image = io.imread(img_path)
        if len(image.shape) < 3:
            image = skimage.color.gray2rgb(image)

        if self.transform is not None:
            # Transforms usually require PIL or a Tensor
            image = Image.fromarray(image)
            image = self.transform(image)

        label = self._y[index].item()
        sens_value = self._sens[index].item()

        # Return them in a format consistent with your 'train(...)' or 'test(...)' functions
        return {"ids": index, "input": image, "label": label, "sens": sens_value}


class FitzpatrickDataModule(BaseDataModule):
    """
    Concrete DataModule for Fitzpatrick images. Inherits from BaseDataModule.
    """

    def __init__(self, config: SharedBaseConfig):
        super().__init__(config)

    # ---------------------
    # BaseDataModule abstract properties
    # ---------------------
    @property
    def X(self) -> torch.Tensor:
        # You can access the underlying dataset via self._base_dataset
        assert self.has_setup, "Need to call setup before accessing X"
        return self._base_dataset.X

    @property
    def y(self) -> torch.Tensor:
        assert self.has_setup, "Need to call setup before accessing y"
        return self._base_dataset.y

    @property
    def sens(self) -> torch.Tensor:
        assert self.has_setup, "Need to call setup before accessing sens"
        if self.name in FAIRNESS_DATASETS:
            return self._base_dataset.sens
        raise NotImplementedError(f"No sensitive attribute for dataset {self.name}.")

    @property
    def labeled_points(self) -> torch.Tensor:
        # E.g., if -1 indicates no label
        labeled_mask = self._base_dataset.y >= 0
        return torch.nonzero(labeled_mask, as_tuple=True)[0]

    @property
    def num_points(self) -> int:
        return len(self._base_dataset)

    @property
    def num_features(self) -> int:
        # For images, you might not rely on 'num_features', but let's return e.g. 3 for color channels
        # or 3*224*224 if you always resize to 224x224
        return 3

    @property
    def num_classes(self) -> int:
        # If labels are 0..N-1
        return len(torch.unique(self._base_dataset.y[self.labeled_points]))

    @property
    def num_sensitive_groups(self) -> int:
        # If 'fitzpatrick_scale' is in 1..6 or 0..5
        return len(torch.unique(self._base_dataset.sens))

    # ---------------------
    # BaseDataModule core methods
    # ---------------------
    def _create_dataset(
        self,
        name: str,
        dataset_dir: str = "",
        /,
        *,
        pred_attrs=[],
        discard_attrs=[],
        sens_attrs=[],
        dataset_args=None,
        force_reprep=False,
    ):
        """
        This method is responsible for creating and returning X, y, sens
        OR returning the actual dataset instance.
        However, in the tabular code, it returns raw (X, y, sens).
        For image data, we can just build the FitzpatrickDataset directly.
        """
        # We'll assume the config has self.config.dataset.csv_file for the CSV
        csv_file = dataset_args.csv_file  # or something from your config
        transform = self._train_transform()

        # Instead of returning (X, y, sens), we can just build FitzpatrickDataset here.
        # Then the _init_with_dataset(...) call can store it in self._base_dataset.
        dataset = FitzpatrickDataset(
            name=name,
            csv_file=csv_file,
            root_dir=dataset_dir,
            args=self.config,
            transform=transform,
        )
        return dataset

    def prepare_data(self) -> None:
        """
        Called once before setup(). You could download data or do big steps here.
        For local data, you might do nothing.
        """
        super().prepare_data()

    def _init_with_dataset(self, dataset: FitzpatrickDataset):
        train_mask, val_mask, calib_mask, test_mask = dataset.process()
        dataset.masks[Stage.TRAIN.mask_dstr] = train_mask
        dataset.masks[Stage.VALIDATION.mask_dstr] = val_mask
        dataset.masks[Stage.CALIBRATION.mask_dstr] = calib_mask
        dataset.masks[Stage.TEST.mask_dstr] = test_mask

        # The tabular example does something like dataset.process(),
        # but we've already done it in the dataset's constructor.
        self._base_dataset = dataset

        # Initialize the dictionary that tracks splits
        # (train, val, test, etc.). We'll build from dataset.masks
        self.split_dict = {
            stage: dataset.get_mask_inds(stage.mask_dstr)
            for stage in Stage
            if stage.mask_dstr in dataset.masks
        }
        self.has_setup = True

    def setup(self, args: SharedBaseConfig = None) -> None:
        """
        Called before the data loaders are created.
        """
        if not self.has_setup:
            dataset = self._create_dataset(
                self.name,
                self.dataset_dir,
                pred_attrs=self.config.dataset.pred_attrs,
                discard_attrs=self.config.dataset.discard_attrs,
                sens_attrs=self.config.dataset.sens_attrs,
                dataset_args=self.config.dataset,
            )
            self._init_with_dataset(dataset)

    # ---------------------
    # Example transforms
    # ---------------------
    def _train_transform(self):
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def _val_transform(self):
        return transforms.Compose(
            [
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

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
