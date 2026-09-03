# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""PASTIS datamodule."""

from collections.abc import Sequence

from functools import partial
from typing import Any

import kornia.augmentation as K

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
        train_folds: Sequence[int] = (1, 2, 3),
        val_folds: Sequence[int] = (4,),
        test_folds: Sequence[int] = (5,),
        padding_length: int = 61,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PASTISDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            train_folds: List of fold indices for training split.
            val_folds: List of fold indices for validation split.
            test_folds: List of fold indices for test split.
            padding_length: Padding length of the time series.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.PASTIS`.
        """
        super().__init__(
            PASTIS, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.train_folds = train_folds
        self.val_folds = val_folds
        self.test_folds = test_folds
        self.padding_length = padding_length
        # Use a picklable callable for multiprocessing DataLoader workers.
        # Local lambdas fail under ``spawn`` with "Can't get local object ...".
        self.collate_fn = partial(
            pad_across_batches, padding_length=self.padding_length
        )

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
        if stage in ['fit']:
            self.train_dataset = PASTIS(folds=self.train_folds, **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = PASTIS(folds=self.val_folds, **self.kwargs)
        if stage in ['test']:
            self.test_dataset = PASTIS(folds=self.test_folds, **self.kwargs)


class PASTIS100DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the PASTIS-R-100 dataset.

    .. versionadded:: 0.9
    """

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        train_folds: Sequence[int] = (1, 2, 3),
        val_folds: Sequence[int] = (4,),
        test_folds: Sequence[int] = (5,),
        padding_length: int = 61,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PASTIS100DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            train_folds: List of fold indices for training split.
            val_folds: List of fold indices for validation split.
            test_folds: List of fold indices for test split  .
            padding_length: Padding length of the time series.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.PASTIS100`.
        """
        super().__init__(
            PASTIS100, batch_size=batch_size, num_workers=num_workers, **kwargs
        )
        self.train_folds = train_folds
        self.val_folds = val_folds
        self.test_folds = test_folds
        self.padding_length = padding_length
        # Use a picklable callable for multiprocessing DataLoader workers.
        # Local lambdas fail under ``spawn`` with "Can't get local object ...".
        self.collate_fn = partial(
            pad_across_batches, padding_length=self.padding_length
        )

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
        if stage in ['fit']:
            self.train_dataset = PASTIS100(folds=self.train_folds, **self.kwargs)
        if stage in ['fit', 'validate']:
            self.val_dataset = PASTIS100(folds=self.val_folds, **self.kwargs)
        if stage in ['test']:
            self.test_dataset = PASTIS100(folds=self.test_folds, **self.kwargs)
