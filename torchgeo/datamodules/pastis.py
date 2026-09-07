# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""PASTIS datamodule."""

import warnings
from collections.abc import Sequence
from functools import partial
from typing import Any

import kornia.augmentation as K

from ..datasets import PASTIS, PASTIS100
from ..datasets.utils import pad_across_batches
from .geo import NonGeoDataModule


class PASTISDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the PASTIS-R dataset.

    .. versionadded:: 0.8
    """

    _dataset_cls: type[PASTIS] = PASTIS

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        train_folds: Sequence[int] = (1, 2, 3),
        val_folds: Sequence[int] = (4,),
        test_folds: Sequence[int] = (5,),
        padding_length: int = 61,
        val_split_pct: float | None = None,
        test_split_pct: float | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize a new PASTISDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: Percentage of the dataset to use as a validation set (Deprecated).
            test_split_pct: Percentage of the dataset to use as a test set (Deprecated).
            train_folds: List of fold indices for training split.
            val_folds: List of fold indices for validation split.
            test_folds: List of fold indices for test split.
            padding_length: Padding length of the time series.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.PASTIS`.

        .. versionadded:: 0.11
            The *train_folds*, *val_folds*, and *test_folds* parameters.
        """

        if val_split_pct is not None or test_split_pct is not None:
            warnings.warn(
                'The val_split_pct and test_split_pct parameters are deprecated and have no effect.'
                'To follow the official PASTIS folds, use train_folds, val_folds, and test_folds instead. ',
                DeprecationWarning,
            )

        super().__init__(
            self._dataset_cls, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        valid_folds = set(range(1, 6))

        for fold, idx in zip(
            ['train', 'val', 'test'], [train_folds, val_folds, test_folds]
        ):
            bad_folds = set(idx) - valid_folds
            assert not bad_folds, (
                f'{fold}_folds have out-of-range indices, got {bad_folds}'
            )

        assert not (set(train_folds) & set(val_folds)), (
            f'train/val overlap: {set(train_folds) & set(val_folds)}'
        )
        assert not (set(train_folds) & set(test_folds)), (
            f'train/test overlap: {set(train_folds) & set(test_folds)}'
        )
        assert not (set(val_folds) & set(test_folds)), (
            f'val/test overlap: {set(val_folds) & set(test_folds)}'
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
            self.train_dataset = self._dataset_cls(
                folds=self.train_folds, **self.kwargs
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = self._dataset_cls(folds=self.val_folds, **self.kwargs)
        if stage in ['test']:
            self.test_dataset = self._dataset_cls(folds=self.test_folds, **self.kwargs)


class PASTIS100DataModule(PASTISDataModule):
    """LightningDataModule implementation for the PASTIS-R-100 dataset.

    .. versionadded:: 0.9
    """

    _dataset_cls: type[PASTIS] = PASTIS100
