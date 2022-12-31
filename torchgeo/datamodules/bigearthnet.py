# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet datamodule."""

from typing import Any, Optional

import torch
from kornia.augmentation import Normalize

from ..datasets import BigEarthNet
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class BigEarthNetDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BigEarthNet dataset.

    Uses the train/val/test splits from the dataset.
    """

    # (VV, VH, B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12)
    # min/max band statistics computed on 100k random samples
    band_mins_raw = torch.tensor(
        [-70.0, -72.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    )
    band_maxs_raw = torch.tensor(
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
    band_mins = torch.tensor(
        [-48.0, -42.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )
    band_maxs = torch.tensor(
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
        """Initialize a LightningDataModule for BigEarthNet based DataLoaders.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.BigEarthNet`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        bands = kwargs.get("bands", "all")
        if bands == "all":
            self.mins = self.band_mins
            self.maxs = self.band_maxs
        elif bands == "s1":
            self.mins = self.band_mins[:2]
            self.maxs = self.band_maxs[:2]
        else:
            self.mins = self.band_mins[2:]
            self.maxs = self.band_maxs[2:]

        self.aug = AugmentationSequential(
            Normalize(mean=self.mins, std=self.maxs - self.mins), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            BigEarthNet(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        self.train_dataset = BigEarthNet(split="train", **self.kwargs)
        self.val_dataset = BigEarthNet(split="val", **self.kwargs)
        self.test_dataset = BigEarthNet(split="test", **self.kwargs)
