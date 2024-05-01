# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""National Agriculture Imagery Program (NAIP) datamodule."""

from typing import Any

import kornia.augmentation as K
from matplotlib.figure import Figure

from ..datasets import NAIP, BoundingBox, Chesapeake13
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class NAIPChesapeakeDataModule(GeoDataModule):
    """LightningDataModule implementation for the NAIP and Chesapeake datasets.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: int | tuple[int, int] = 256,
        length: int | None = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new NAIPChesapeakeDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.NAIP` (prefix keys with ``naip_``) and
                :class:`~torchgeo.datasets.Chesapeake13`
                (prefix keys with ``chesapeake_``).
        """
        self.naip_kwargs = {}
        self.chesapeake_kwargs = {}
        for key, val in kwargs.items():
            if key.startswith("naip_"):
                self.naip_kwargs[key[5:]] = val
            elif key.startswith("chesapeake_"):
                self.chesapeake_kwargs[key[11:]] = val

        super().__init__(
            Chesapeake13,
            batch_size,
            patch_size,
            length,
            num_workers,
            **self.chesapeake_kwargs,
        )

        self.aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=["image", "mask"]
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.chesapeake = Chesapeake13(**self.chesapeake_kwargs)
        self.naip = NAIP(**self.naip_kwargs)
        self.dataset = self.chesapeake & self.naip

        roi = self.dataset.bounds
        midx = roi.minx + (roi.maxx - roi.minx) / 2
        midy = roi.miny + (roi.maxy - roi.miny) / 2

        if stage in ["fit"]:
            train_roi = BoundingBox(
                roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt
            )
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.dataset, self.patch_size, self.batch_size, self.length, train_roi
            )
        if stage in ["fit", "validate"]:
            val_roi = BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
            self.val_sampler = GridGeoSampler(
                self.dataset, self.patch_size, self.patch_size, val_roi
            )
        if stage in ["test"]:
            test_roi = BoundingBox(
                roi.minx, roi.maxx, midy, roi.maxy, roi.mint, roi.maxt
            )
            self.test_sampler = GridGeoSampler(
                self.dataset, self.patch_size, self.patch_size, test_roi
            )

    def plot(self, *args: Any, **kwargs: Any) -> Figure:
        """Run NAIP plot method.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A matplotlib Figure with the image, ground truth, and predictions.

        .. versionadded:: 0.4
        """
        return self.naip.plot(*args, **kwargs)
