# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""USAVars datamodule."""

from typing import Any, Callable, Dict, Optional, Sequence

import pytorch_lightning as pl
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from ..datasets import USAVars
from .utils import dataset_split


class USAVarsDataModule(pl.LightningModule):
    """LightningDataModule implementation for the USAVars dataset.

    Finish this.

    .. versionadded:: 0.3
    """

    def __init__(
        self,
        root_dir: str,
        labels: Sequence[str] = USAVars.ALL_LABELS,
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        fixed_shuffle: bool = False,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.1,
    ) -> None:
        """Initialize a LightningDataModule for USAVars based DataLoaders.

        Args:
            root_dir: The root argument passed to the USAVars Dataset classes
            labels: The labels argument passed to the USAVars Dataset classes
            transforms: a function/transform that takes input sample and its target as
                            entry and returns a transformed version
            fixed_shuffle: always shuffles DataLoader to same order
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            val_split_pct: What percentage of the dataset to use as a validation set
            test_split_pct: What percentage of the dataset to use as a test set
        """
        super().__init__()
        self.root_dir = root_dir
        self.labels = labels
        self.transforms = transforms
        self.fixed_shuffle = fixed_shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        assert val_split_pct + test_split_pct <= 1.0
        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        if fixed_shuffle:
            torch.manual_seed(0)

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        USAVars(self.root_dir, self.labels, checksum=False)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.
        """
        dataset = USAVars(self.root_dir, self.labels, transforms=self.transforms)
        self.train_dataset, self.val_dataset, self.test_dataset = dataset_split(
            dataset, val_pct=self.val_split_pct, test_pct=self.test_split_pct
        )

    def train_dataloader(self) -> DataLoader[Any]:
        """Return a DataLoader for training."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
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
