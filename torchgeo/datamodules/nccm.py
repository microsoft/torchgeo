# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Northeastern China Crop Map (NCCM) datamodule."""

from typing import Any, Optional, Union

import kornia.augmentation as K
from matplotlib.figure import Figure

from ..datasets import NCCM, BoundingBox, Sentinel2
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class NCCMSentinel2DataModule(GeoDataModule):
    """LightningDataModule implementation for the NCCM and Sentinel2 datasets.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 256,
        length: Optional[int] = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new NCCMSentinel2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.NCCM` (prefix keys with ``nccm_``) and
                :class:`~torchgeo.datasets.Sentinel2`
                (prefix keys with ``sentinel2_``).
        """
        self.nccm_kwargs = {}
        self.sentinel2_kwargs = {}
        for key, val in kwargs.items():
            if key.startswith("nccm_"):
                self.nccm_kwargs[key[5:]] = val
            elif key.startswith("sentinel2_"):
                self.sentinel2_kwargs[key[10:]] = val

        super().__init__(
            NCCM, batch_size, patch_size, length, num_workers, **self.nccm_kwargs
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
        self.nccm = NCCM(**self.nccm_kwargs)
        self.dataset = self.sentinel2 & self.nccm

        roi = self.dataset.bounds
        print("roi is", roi)
        midx = roi.minx + (roi.maxx - roi.minx) / 2
        midy = roi.miny + (roi.maxy - roi.miny) / 2
        print("midx is ", midx)
        print("midy is", midy)


        if stage in ["fit"]:
            print("train roi all parameters ", roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
            train_roi = BoundingBox(
                roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt
            )

          

            print("train batch sampler ", self.patch_size, self.batch_size,self.length)
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
        """Run NCCM plot method.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A matplotlib Figure with the image, ground truth, and predictions.

        .. versionadded:: 0.4
        """
        return self.nccm.plot(*args, **kwargs)
