# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import OSCD
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from ..transforms.transforms import _RandomNCrop
from .geo import NonGeoDataModule
from .utils import dataset_split

MEAN = {
    "B01": 1583.0741,
    "B02": 1374.3202,
    "B03": 1294.1616,
    "B04": 1325.6158,
    "B05": 1478.7408,
    "B06": 1933.0822,
    "B07": 2166.0608,
    "B08": 2076.4868,
    "B8A": 2306.0652,
    "B09": 690.9814,
    "B10": 16.2360,
    "B11": 2080.3347,
    "B12": 1524.6930,
}

STD = {
    "B01": 52.1937,
    "B02": 83.4168,
    "B03": 105.6966,
    "B04": 151.1401,
    "B05": 147.4615,
    "B06": 115.9289,
    "B07": 123.1974,
    "B08": 114.6483,
    "B8A": 141.4530,
    "B09": 73.2758,
    "B10": 4.8368,
    "B11": 213.4821,
    "B12": 179.4793,
}


class OSCDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the OSCD dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: tuple[int, int] | int = 64,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new OSCDDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.OSCD`.
        """
        super().__init__(OSCD, 1, num_workers, **kwargs)

        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.bands = kwargs.get("bands", OSCD.all_bands)
        self.mean = torch.tensor([MEAN[b] for b in self.bands])
        self.std = torch.tensor([STD[b] for b in self.bands])

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(self.patch_size, batch_size),
            data_keys=["image1", "image2", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            self.dataset = OSCD(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset = dataset_split(
                self.dataset, val_pct=self.val_split_pct
            )
        if stage in ["test"]:
            self.test_dataset = OSCD(split="test", **self.kwargs)
