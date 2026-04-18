# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""xBD datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from torch.utils.data import random_split
from typing_extensions import deprecated

from ..datasets import xBD
from .geo import NonGeoDataModule


class xBDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the xBD dataset.

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
        """Initialize a new xBDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            val_split_pct: What percentage of the dataset to use as a validation set
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.xBD`.
        """
        super().__init__(xBD, batch_size, num_workers, **kwargs)
        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
            data_keys=None,
            keepdim=True,
        )
        self.val_split_pct = val_split_pct

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            self.dataset = xBD(split='train', **self.kwargs)
            generator = torch.Generator().manual_seed(0)
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [1 - self.val_split_pct, self.val_split_pct], generator
            )
        if stage in ['test']:
            self.test_dataset = xBD(split='test', **self.kwargs)


@deprecated('Use torchgeo.datamodules.xBDDataModule instead')
class XView2DataModule(xBDDataModule):
    """Deprecated alias for the xBD datamodule."""
