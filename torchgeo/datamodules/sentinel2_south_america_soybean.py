# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


"""South America Soybean datamodule."""


from typing import Any

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from matplotlib.figure import Figure

from ..datasets import Sentinel2, SouthAmericaSoybean, random_bbox_assignment
from ..samplers import GridGeoSampler, RandomGeoSampler
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class Sentinel2SouthAmericaSoybeanDataModule(GeoDataModule):
    """LightningDataModule for SouthAmericaSoybean and Sentinel2 datasets.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 64,
        length: int | None = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new Sentinel2SouthAmericaSoybeanDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SouthAmericaSoybean`
                (prefix keys with ``south_america_soybean_``) and
                :class:`~torchgeo.datasets.Sentinel2`
                (prefix keys with ``sentinel2_``).
        """
        self.south_america_soybean_kwargs = {}
        self.sentinel2_kwargs = {}
        for key, val in kwargs.items():
            if key.startswith("south_america_soybean_"):
                self.south_america_soybean_kwargs[key[22:]] = val
            elif key.startswith("sentinel2_"):
                self.sentinel2_kwargs[key[10:]] = val

        super().__init__(
            SouthAmericaSoybean,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )

        self.train_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.RandomResizedCrop(_to_tuple(self.patch_size), scale=(0.6, 1.0)),
            K.RandomVerticalFlip(p=0.5),
            K.RandomHorizontalFlip(p=0.5),
            data_keys=["image", "mask"],
            extra_args={
                DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": None}
            },
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.sentinel2 = Sentinel2(**self.sentinel2_kwargs)
        self.south_america_soybean = SouthAmericaSoybean(
            **self.south_america_soybean_kwargs
        )
        self.dataset = self.sentinel2 & self.south_america_soybean

        generator = torch.Generator().manual_seed(1)
        (self.train_dataset, self.val_dataset, self.test_dataset) = (
            random_bbox_assignment(self.dataset, [0.8, 0.1, 0.1], generator=generator)
        )

        if stage in ["fit"]:
            self.train_sampler = RandomGeoSampler(
                self.train_dataset, self.patch_size, self.length
            )
        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ["test"]:
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )

    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        """Run SouthAmericaSoybean plot method.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A matplotlib Figure with the image, ground truth, and predictions.
        """
        return self.south_america_soybean.plot(*args, **kwargs)
