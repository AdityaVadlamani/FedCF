from collections import Counter
from math import isclose
from typing import Any, List, Optional, Tuple
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split
import torch
from transformers import AutoTokenizer

from conformal_fairness.constants import CAIL
from conformal_fairness.config import SharedBaseConfig
from conformal_fairness.data.tabular_datamodule import TabularDataModule, TabularDataset


class FairlexDataset(TabularDataset):
    """
    Concrete dataset class for folktables datasets (https://huggingface.co/datasets/coastalcph/fairlex)
    """

    def __init__(self, name: str, args: SharedBaseConfig):
        """
        Args:
            name (str): Dataset name.
            args (SharedBaseConfig): Shared config (contains .seed, .dataset_split_fractions, etc.).
            transform: Optional transform to apply to each image.
        """

        if name not in (CAIL,):
            raise ValueError("Invalid dataset provided for FairlexDataset")

        dataset: DatasetDict = load_dataset(
            "coastalcph/fairlex",
            name,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        model_name = f"coastalcph/fairlex-{name}-minilm"

        X = []
        y = []
        sens = []

        for _, data_split in dataset.items():
            for elem in data_split:
                X.append(elem["text"])
                y.append(elem["label"])
                sens_attrs_str = ",".join(args.dataset.sens_attrs)
                if sens_attrs_str == "gender":
                    sens.append(elem["defendant_gender"])
                elif sens_attrs_str == "region":
                    sens.append(elem["court_region"])
                else:
                    sens.append(7 * elem["defendant_gender"] + elem["court_region"])

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, force_download=True
            )

        # Apply tokenization
        X = [self._tokenize_text(text) for text in X]

        super(FairlexDataset, self).__init__(
            name=name,
            X=X,
            y=torch.tensor(y).reshape((-1,)),
            sens=torch.tensor(sens).reshape((-1,)),
            args=args,
        )

    def _tokenize_text(self, text):
        """
        Tokenizes and pads/truncates the input text.
        """
        tokenized = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            max_length=512,  # Max length for minilm models
            truncation=True,
        )

        return {
            "input_ids": tokenized["input_ids"].squeeze(0),
            "attention_mask": tokenized["attention_mask"].squeeze(0),
        }

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.item()

        input_data = self.X[index]
        return {
            "ids": torch.tensor(index),
            "input_ids": input_data["input_ids"],
            "attention_mask": input_data["attention_mask"],
            "label": self.y[index],
            "sens": self.sens[index],
        }

    def process(self):
        return self._setup_masks(len(self.X))

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

        groups, labels = self.sens[labeled_points], self.y[labeled_points]
        group_label_pairs = list(zip(groups.tolist(), labels.tolist()))

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


class FairlexDatamodule(TabularDataModule):
    """
    Concrete DataModule for Fairlex dataset.
    """

    def __init__(
        self,
        config: SharedBaseConfig,
    ):
        super().__init__(config)

    @property
    def X(self) -> List[Any]:
        assert self.has_setup, "Need to call setup before accessing properties"
        return self._base_dataset.X

    @property
    def num_points(self) -> int:
        assert self.has_setup, "Need to call setup before accessing properties"
        return len(self.X)

    @property
    def num_features(self):
        return 1

    @property
    def num_classes(self):
        if self.name == CAIL:
            return 6
        else:
            raise ValueError("Invalid dataset provided for FairlexDatamodule")

    @property
    def num_sensitive_groups(self) -> int:
        assert self.has_setup, "Need to call setup before accessing properties"
        if self.name == CAIL:
            sens_attrs_str = ",".join(self.config.dataset.sens_attrs)
            if sens_attrs_str == "gender":
                return 2
            elif sens_attrs_str == "region":
                return 7
            else:
                return 14

        raise NotImplementedError(
            f"No sensitive groups in {self.name} to be considered"
        )

    def _create_dataset(
        self,
        name: str,
    ):
        dataset = FairlexDataset(
            name=name,
            args=self.config,
        )
        return dataset

    def prepare_data(self):
        pass

    def setup(self, args: SharedBaseConfig = None) -> None:
        del args  # Unused
        if not self.has_setup:
            dataset = self._create_dataset(
                self.name,
            )
            self._init_with_dataset(dataset)
