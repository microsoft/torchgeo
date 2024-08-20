# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""xView2 datamodule."""

from typing import Any

import torch
from torch.utils.data import random_split

from ..datasets import XView2
from .geo import NonGeoDataModule


class XView2DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the xView2 dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new XView2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: What percentage of the dataset to use as a validation set
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.XView2`.
        """
        super().__init__(XView2, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            self.dataset = XView2(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
        if stage in ['test']:
            self.test_dataset = XView2(split='test', **self.kwargs)
