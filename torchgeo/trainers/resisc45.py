# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""RESISC45 trainer."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize

from ..datasets import RESISC45
from ..datasets.utils import dataset_split
from .tasks import ClassificationTask

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class RESISC45ClassificationTask(ClassificationTask):
    """LightningModule for training models on the RESISC45 Dataset."""

    num_classes = 45


class RESISC45DataModule(pl.LightningDataModule):
    """LightningDataModule implementation for the RESISC45 dataset.

    Uses the train/val/test splits from the dataset.
    """

    band_means = torch.tensor(  # type: ignore[attr-defined]
        [0.36801773, 0.38097873, 0.343583]
    )

    band_stds = torch.tensor(  # type: ignore[attr-defined]
        [0.14540215, 0.13558227, 0.13203649]
    )

    def __init__(
        self,
        root_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        weights: str = "random",
        unsupervised_mode: bool = False,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a LightningDataModule for RESISC45 based DataLoaders.

        Args:
            root_dir: The ``root`` arugment to pass to the RESISC45 Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            weights: Either "random", "imagenet_only", "imagenet_and_random", or
                "random_rgb"
            unsupervised_mode: Makes the train dataloader return imagery from the train,
                val, and test sets
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
        """
        super().__init__()  # type: ignore[no-untyped-call]
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.weights = weights
        self.unsupervised_mode = unsupervised_mode

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        self.norm = Normalize(self.band_means, self.band_stds)

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: input image dictionary

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        sample["image"] = self.norm(sample["image"])
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        RESISC45(self.root_dir, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main ``Dataset`` objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        transforms = Compose([self.preprocess])

        if not self.unsupervised_mode:

            dataset = RESISC45(self.root_dir, transforms=transforms)
            self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
                dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
            )
        else:

            self.train_dataset = RESISC45(self.root_dir, transforms=transforms)
            self.val_dataset, self.test_dataset = None, None  # type: ignore[assignment]

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
        if self.unsupervised_mode or self.val_split_pct == 0:
            return self.train_dataloader()
        else:
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
        if self.unsupervised_mode or self.test_split_pct == 0:
            return self.train_dataloader()
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=False,
            )
