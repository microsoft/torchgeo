# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""USAVars datamodule."""

from typing import Any, Dict, Optional, Sequence

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from ..datasets import USAVars


class USAVarsDataModule(pl.LightningModule):
    """LightningDataModule implementation for the USAVars dataset.

    Uses random train/val/test splits.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        root_dir: str,
        labels: Sequence[str] = USAVars.ALL_LABELS,
        batch_size: int = 64,
        num_workers: int = 0,
    ) -> None:
        """Initialize a LightningDataModule for USAVars based DataLoaders.

        Args:
            root_dir: The root argument passed to the USAVars Dataset classes
            labels: The labels argument passed to the USAVars Dataset classes
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
        """
        super().__init__()
        self.root_dir = root_dir
        self.labels = labels
        self.batch_size = batch_size
        self.num_workers = num_workers

    def preprocess(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single sample from the Dataset.

        Args:
            sample: dictionary containing image

        Returns:
            preprocessed sample
        """
        sample["image"] = sample["image"].float()
        sample["image"] /= 255.0
        return sample

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        USAVars(self.root_dir, labels=self.labels, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.
        """
        self.train_dataset = USAVars(
            self.root_dir, "train", self.labels, transforms=self.preprocess
        )
        self.val_dataset = USAVars(
            self.root_dir, "val", self.labels, transforms=self.preprocess
        )
        self.test_dataset = USAVars(
            self.root_dir, "test", self.labels, transforms=self.preprocess
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for validation."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for testing."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
