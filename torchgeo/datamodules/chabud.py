# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ChaBuD datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import ChaBuD
from .geo import NonGeoDataModule


class ChaBuDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the ChaBuD dataset.

    Uses the train/val splits from the dataset

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
        """Initialize a new ChaBuDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ChaBuD`.
        """
        bands = kwargs.get('bands', ChaBuD.all_bands)
        band_indices = [ChaBuD.all_bands.index(b) for b in bands]
        mins = self.min[band_indices]
        maxs = self.max[band_indices]

        self.mean = mins
        self.std = maxs - mins

        super().__init__(ChaBuD, batch_size, num_workers, **kwargs)
        self.aug = K.AugmentationSequential(
            K.VideoSequential(K.Normalize(mean=self.mean, std=self.std)),
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
            self.train_dataset = ChaBuD(split='train', **self.kwargs)
            self.val_dataset = ChaBuD(split='val', **self.kwargs)
