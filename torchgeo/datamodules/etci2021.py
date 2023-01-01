# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ETCI 2021 datamodule."""

from typing import Any, Optional

import kornia.augmentation as K
from torch.utils.data import random_split

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
        """Initialize a new LightningDataModule instance.

        Args:
            batch_size: The batch size to use in all created DataLoaders
            num_workers: The number of workers to use in all created DataLoaders
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.ETCI2021`
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.kwargs = kwargs

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.band_means, std=self.band_stds), data_keys=["image"]
        )

    def prepare_data(self) -> None:
        """Make sure that the dataset is downloaded.

        This method is only called once per run.
        """
        if self.kwargs.get("download", False):
            ETCI2021(**self.kwargs)

    def setup(self, stage: Optional[str] = None) -> None:
        """Initialize the main Dataset objects.

        This method is called once per GPU per run.

        Args:
            stage: stage to set up
        """
        train_val_dataset = ETCI2021(split="train", **self.kwargs)
        self.test_dataset = ETCI2021(split="val", **self.kwargs)

        size_train_val = len(train_val_dataset)
        size_train = round(0.8 * size_train_val)
        size_val = size_train_val - size_train

        self.train_dataset, self.val_dataset = random_split(
            train_val_dataset, [size_train, size_val]
        )
