# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""PatternNet datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import PatternNet
from .geo import NonGeoDataModule


class PatternNetDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the PatternNet dataset.

    Uses random train/val/test splits.

    .. versionadded:: 0.8
    """

    mean = torch.tensor([91.48, 91.78, 81.23])
    std = torch.tensor([49.74, 47.18, 45.43])

    def __init__(
        self,
        batch_size: int = 64,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PatternNetDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Fraction of dataset to use for validation.
            test_split_pct: Fraction of dataset to use for testing.
            **kwargs: Additional keyword arguments passed to :class:`~torchgeo.datasets.PatternNet`.
        """
        super().__init__(PatternNet, batch_size, num_workers, **kwargs)

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.Resize(size=256),
            data_keys=None,
            keepdim=True,
        )
        self.train_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.Resize(size=256),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        dataset = PatternNet(**self.kwargs)

        generator = torch.Generator().manual_seed(0)
        train_spilt_pct = 1 - self.val_split_pct - self.test_split_pct
        lengths = [train_spilt_pct, self.val_split_pct, self.test_split_pct]
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            dataset, lengths, generator
        )
