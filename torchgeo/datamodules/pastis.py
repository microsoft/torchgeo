# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""PASTIS datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import PASTIS, PASTIS100
from ..datasets.utils import pad_across_batches
from .geo import NonGeoDataModule


class PASTISDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the PASTIS dataset.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        padding_length: int = 61,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PASTISDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            padding_length: Padding length of the time series.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.PASTIS`.
        """
        super().__init__(
            PASTIS, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.padding_length = padding_length
        self.collate_fn = lambda batch: pad_across_batches(
            batch, padding_length=self.padding_length
        )

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = PASTIS(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [
                1 - self.val_split_pct - self.test_split_pct,
                self.val_split_pct,
                self.test_split_pct,
            ],
            generator,
        )


class PASTIS100DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the PASTIS-R-100 dataset.

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        val_split_pct: float = 0.2,
        test_split_pct: float = 0.2,
        padding_length: int = 61,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PASTIS100DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set.
            test_split_pct: Percentage of the dataset to use as a test set.
            padding_length: Padding length of the time series.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.PASTIS100`.
        """
        super().__init__(
            PASTIS100, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.padding_length = padding_length
        self.collate_fn = lambda batch: pad_across_batches(
            batch, padding_length=self.padding_length
        )

        self.val_split_pct = val_split_pct
        self.test_split_pct = test_split_pct
        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = PASTIS100(**self.kwargs)
        generator = torch.Generator().manual_seed(0)
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            self.dataset,
            [
                1 - self.val_split_pct - self.test_split_pct,
                self.val_split_pct,
                self.test_split_pct,
            ],
            generator,
        )
