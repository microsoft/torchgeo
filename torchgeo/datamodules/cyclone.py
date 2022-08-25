# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tropical Cyclone Wind Estimation Competition datamodule."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

from ..datasets import TropicalCycloneWindEstimation

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class CycloneDataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the NASA Cyclone dataset.

    Implements 80/20 train/val splits based on hurricane storm ids.
    See :func:`setup` for more details.
    """

    def __init__(
        self,
        root_dir: str,
        seed: int,
        batch_size: int = 64,
        num_workers: int = 0,
        api_key: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for NASA Cyclone based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the
                TropicalCycloneWindEstimation Datasets classes
            seed: The seed value to use when doing the sklearn based GroupShuffleSplit
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            api_key: The RadiantEarth MLHub API key to use if the dataset needs to be
                downloaded
        """
        super().__init__()
        self.root_dir = root_dir
        self.seed = seed
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.api_key = api_key

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image and target

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        sample["image"] = (
            sample["image"].unsqueeze(0).repeat(3, 1, 1)
        )  # convert from grayscale to 3 channel
        if "label" in sample:
            sample["label"] = torch.as_tensor(sample["label"]).float()

        return sample

    def prepare_data(self) -> None:
        """Initialize the main ``Dataset`` objects for use in :func:`setup`.

        This includes optionally downloading the dataset. This is done once per node,
        while :func:`setup` is done once per GPU.
        """
        TropicalCycloneWindEstimation(
            self.root_dir,
            split="train",
            download=self.api_key is not None,
            api_key=self.api_key,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        """Create the train/val/test splits based on the original Dataset objects.

        The splits should be done here vs. in :func:`__init__` per the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html#setup.

        We split samples between train/val by the ``storm_id`` property. I.e. all
        samples with the same ``storm_id`` value will be either in the train or the val
        split. This is important to test one type of generalizability -- given a new
        storm, can we predict its windspeed. The test set, however, contains *some*
        storms from the training set (specifically, the latter parts of the storms) as
        well as some novel storms.

        Args:
            stage: stage to set up
        """
        self.all_train_dataset = TropicalCycloneWindEstimation(
            self.root_dir, split="train", transforms=self.preprocess, download=False
        )

        self.all_test_dataset = TropicalCycloneWindEstimation(
            self.root_dir, split="test", transforms=self.preprocess, download=False
        )

        storm_ids = []
        for item in self.all_train_dataset.collection:
            storm_id = item["href"].split("/")[0].split("_")[-2]
            storm_ids.append(storm_id)

        train_indices, val_indices = next(
            GroupShuffleSplit(test_size=0.2, n_splits=2, random_state=self.seed).split(
                storm_ids, groups=storm_ids
            )
        )

        self.train_dataset = Subset(self.all_train_dataset, train_indices)
        self.val_dataset = Subset(self.all_train_dataset, val_indices)
        self.test_dataset = Subset(
            self.all_test_dataset, range(len(self.all_test_dataset))
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training.

        Returns:
            training data loader
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation.

        Returns:
            validation data loader
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing.

        Returns:
            testing data loader
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
