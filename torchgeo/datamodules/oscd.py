# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""OSCD datamodule."""

from typing import Any, Tuple, Union

import kornia.augmentation as K
import torch
from einops import repeat

from ..datasets import OSCD
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from ..transforms.transforms import _ExtractTensorPatches, _RandomNCrop
from .geo import NonGeoDataModule
from .utils import dataset_split


class OSCDDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the OSCD dataset.

    Uses the train/test splits from the dataset and further splits
    the train split into train/val splits.

    .. versionadded:: 0.2
    """

    mean = torch.tensor(
        [
            1583.0741,
            1374.3202,
            1294.1616,
            1325.6158,
            1478.7408,
            1933.0822,
            2166.0608,
            2076.4868,
            2306.0652,
            690.9814,
            16.2360,
            2080.3347,
            1524.6930,
        ]
    )

    std = torch.tensor(
        [
            52.1937,
            83.4168,
            105.6966,
            151.1401,
            147.4615,
            115.9289,
            123.1974,
            114.6483,
            141.4530,
            73.2758,
            4.8368,
            213.4821,
            179.4793,
        ]
    )

    def __init__(
        self,
        num_tiles_per_batch: int = 16,
        num_patches_per_tile: int = 16,
        patch_size: Union[Tuple[int, int], int] = 64,
        val_split_pct: float = 0.2,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new OSCDDataModule instance.

        The OSCD dataset contains images that are too large to pass
        directly through a model. Instead, we randomly sample patches from image tiles
        during training and chop up image tiles into patch grids during evaluation.
        During training, the effective batch size is equal to
        ``num_tiles_per_batch`` x ``num_patches_per_tile``.

        Args:
            num_tiles_per_batch: Number of image tiles to sample from during training.
            num_patches_per_tile: Number of patches to randomly sample from each image
                tile during training.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
                Should be a multiple of 32 for most segmentation architectures.
            val_split_pct: Percentage of the dataset to use as a validation set.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.OSCD`.
        """
        super().__init__(OSCD, 1, num_workers, **kwargs)

        self.train_batch_size = num_tiles_per_batch
        self.num_patches_per_tile = num_patches_per_tile
        self.patch_size = _to_tuple(patch_size)
        self.val_split_pct = val_split_pct

        self.bands = kwargs.get("bands", "all")
        if self.bands == "rgb":
            self.mean = self.mean[[3, 2, 1]]
            self.std = self.std[[3, 2, 1]]

        # Change detection, 2 images from different times
        self.mean = repeat(self.mean, "c -> (t c)", t=2)
        self.std = repeat(self.std, "c -> (t c)", t=2)

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _RandomNCrop(self.patch_size, self.num_patches_per_tile),
            data_keys=["image", "mask"],
        )
        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            _ExtractTensorPatches(self.patch_size),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        if stage in ["fit", "validate"]:
            dataset = OSCD(split="train", **self.kwargs)
            self.train_dataset, self.val_dataset = dataset_split(
                dataset, val_pct=self.val_split_pct
            )
        elif stage in ["test"]:
            self.test_dataset = OSCD(split="test", **self.kwargs)
