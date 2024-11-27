# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tropical Cyclone Wind Estimation Competition datamodule."""

from typing import Any

from torch.utils.data import Subset

from ..datasets import TropicalCyclone
from .geo import NonGeoDataModule
from .utils import group_shuffle_split


class TropicalCycloneDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the NASA Cyclone dataset.

    Implements 80/20 train/val splits based on hurricane storm ids.
    See :func:`setup` for more details.

    .. versionchanged:: 0.4
        Class name changed from CycloneDataModule to TropicalCycloneDataModule to be
        consistent with TropicalCyclone dataset.
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new TropicalCycloneDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.TropicalCyclone`.
        """
        super().__init__(TropicalCyclone, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ['fit', 'validate']:
            dataset = TropicalCyclone(split='train', **self.kwargs)
            train_indices, val_indices = group_shuffle_split(
                dataset.features['Storm ID'], test_size=0.2, random_state=0
            )
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
        if stage in ['test']:
            self.test_dataset = TropicalCyclone(split='test', **self.kwargs)
