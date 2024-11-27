# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet datamodule."""

from typing import Any

import torch

from ..datasets import BigEarthNet
from .geo import NonGeoDataModule


class BigEarthNetDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BigEarthNet dataset.

    Uses the train/val/test splits from the dataset.
    """

    # (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    # min/max band statistics computed on 100k random samples
    mins_raw = torch.tensor(
        [-70.0, -72.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    maxs_raw = torch.tensor(
        [
            31.0,
            35.0,
            18556.0,
            20528.0,
            18976.0,
            17874.0,
            16611.0,
            16512.0,
            16394.0,
            16672.0,
            16141.0,
            16097.0,
            15336.0,
            15203.0,
        ]
    )

    # min/max band statistics computed by percentile clipping the
    # above to samples to [2, 98]
    mins = torch.tensor(
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    maxs = torch.tensor(
        [
            6.0,
            16.0,
            9859.0,
            12872.0,
            13163.0,
            14445.0,
            12477.0,
            12563.0,
            12289.0,
            15596.0,
            12183.0,
            9458.0,
            5897.0,
            5544.0,
        ]
    )

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new BigEarthNetDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.BigEarthNet`.
        """
        bands = kwargs.get('bands', 'all')
        if bands == 'all':
            mins = self.mins
            maxs = self.maxs
        elif bands == 's1':
            mins = self.mins[:2]
            maxs = self.maxs[:2]
        else:
            mins = self.mins[2:]
            maxs = self.maxs[2:]
        self.mean = mins
        self.std = maxs - mins

        super().__init__(BigEarthNet, batch_size, num_workers, **kwargs)
