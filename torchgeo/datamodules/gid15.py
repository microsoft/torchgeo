# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""GID-15 datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split

from ..datasets import GID15
from ..samplers.utils import _to_tuple
from ..transforms.transforms import _ExtractPatches
from .geo import NonGeoDataModule


class GID15DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the GID-15 dataset.

    Uses the train/test splits from the dataset.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: tuple[int, int] | int = 64,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new GID15DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.GID15`.
        """
        super().__init__(
            GID15, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.train_aug = self.val_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomCrop(self.patch_size, pad_if_needed=True),
            data_keys=None,
            keepdim=True,
        )
        self.predict_aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractPatches(window_size=self.patch_size),
            data_keys=None,
            keepdim=True,
            same_on_batch=True,
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            self.dataset = GID15(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
        if stage in ['test']:
            # Test set masks are not public, use for prediction instead
            self.predict_dataset = GID15(split='test', **self.kwargs)
