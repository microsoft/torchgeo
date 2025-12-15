# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Cloud Cover Detection Challenge datamodule."""

from typing import Any

import torch
from torch.utils.data import random_split

from ..datasets import CloudCoverDetection
from .geo import NonGeoDataModule


class CloudCoverDetectionDataModule(NonGeoDataModule):
    """LightningDataModule implementation for Cloud Cover Detection.

    Splits the training split into train/val subsets using ``val_split_pct``.

    .. versionadded:: 0.9
    """

    mean = torch.tensor(0.0)
    std = torch.tensor(10000.0)

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new CloudCoverDetectionDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the training data to reserve for validation.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CloudCoverDetection`.
        """
        super().__init__(CloudCoverDetection, batch_size, num_workers, **kwargs)
        self.val_split_pct = val_split_pct

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            self.dataset = CloudCoverDetection(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )

        if stage in ['test']:
            self.test_dataset = CloudCoverDetection(split='test', **self.kwargs)
