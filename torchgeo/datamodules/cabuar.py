# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CaBuAr datamodule."""

from typing import Any

import torch
from einops import repeat

from ..datasets import CaBuAr
from .geo import NonGeoDataModule


class CaBuArDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the CaBuAr dataset.

    Uses the train/val/test splits from the dataset

    .. versionadded:: 0.6
    """

    # min/max values computed on train set using 2/98 percentiles
    min = torch.tensor(
        [0.0, 1.0, 73.0, 39.0, 46.0, 25.0, 26.0, 21.0, 17.0, 1.0, 20.0, 21.0]
    )
    max = torch.tensor(
        [
            1926.0,
            2174.0,
            2527.0,
            2950.0,
            3237.0,
            3717.0,
            4087.0,
            4271.0,
            4290.0,
            4219.0,
            4568.0,
            3753.0,
        ]
    )

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new CaBuArDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CaBuAr`.
        """
        bands = kwargs.get('bands', CaBuAr.all_bands)
        band_indices = [CaBuAr.all_bands.index(b) for b in bands]
        mins = self.min[band_indices]
        maxs = self.max[band_indices]

        # Change detection, 2 images from different times
        mins = repeat(mins, 'c -> (t c)', t=2)
        maxs = repeat(maxs, 'c -> (t c)', t=2)

        self.mean = mins
        self.std = maxs - mins

        super().__init__(CaBuAr, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test'.
        """
        if stage in ['fit', 'validate']:
            self.train_dataset = CaBuAr(split='train', **self.kwargs)
            self.val_dataset = CaBuAr(split='val', **self.kwargs)
        elif stage == 'test':
            self.test_dataset = CaBuAr(split='test', **self.kwargs)
