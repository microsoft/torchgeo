# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ETCI 2021 datamodule."""

from typing import Any

import kornia.augmentation as K

from ..datasets import ETCI2021
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class ETCI2021DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the ETCI2021 dataset.

    Splits the existing train split from the dataset into train/val with 80/20
    proportions, then uses the existing val dataset as the test data.

    .. versionadded:: 0.2
    """

    band_means = [
        128.02253931,
        128.02253931,
        128.02253931,
        128.11221701,
        128.11221701,
        128.11221701,
    ]
    band_stds = [89.8145088, 89.8145088, 89.8145088, 95.2797861, 95.2797861, 95.2797861]

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new ETCI2021DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ETCI2021`.
        """
        super().__init__(ETCI2021, batch_size, num_workers, **kwargs)

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.band_means, std=self.band_stds), data_keys=["image"]
        )
