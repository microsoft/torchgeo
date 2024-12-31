# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""National Agriculture Imagery Program (NAIP) datamodule."""

from typing import Any

import kornia.augmentation as K
from matplotlib.figure import Figure

from ..datasets import (
    NAIP,
    BoundingBox,
    ChesapeakeDC,
    ChesapeakeDE,
    ChesapeakeMD,
    ChesapeakeNY,
    ChesapeakePA,
    ChesapeakeVA,
    ChesapeakeWV,
)
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
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
                :class:`~torchgeo.datasets.Chesapeake`
                (prefix keys with ``chesapeake_``).
        """
        self.naip_kwargs = {}
        self.chesapeake_kwargs = {}
        for key, val in kwargs.items():
            if key.startswith('naip_'):
                self.naip_kwargs[key[5:]] = val
            elif key.startswith('chesapeake_'):
                self.chesapeake_kwargs[key[11:]] = val

        super().__init__(
            NAIP, batch_size, patch_size, length, num_workers, **self.naip_kwargs
        )

        self.aug = K.AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std), data_keys=None, keepdim=True
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.naip = NAIP(**self.naip_kwargs)
        dc = ChesapeakeDC(**self.chesapeake_kwargs)
        de = ChesapeakeDE(**self.chesapeake_kwargs)
        md = ChesapeakeMD(**self.chesapeake_kwargs)
        ny = ChesapeakeNY(**self.chesapeake_kwargs)
        pa = ChesapeakePA(**self.chesapeake_kwargs)
        va = ChesapeakeVA(**self.chesapeake_kwargs)
        wv = ChesapeakeWV(**self.chesapeake_kwargs)
        self.chesapeake = dc | de | md | ny | pa | va | wv
        self.dataset = self.naip & self.chesapeake

        roi = self.dataset.bounds
        midx = roi.minx + (roi.maxx - roi.minx) / 2
        midy = roi.miny + (roi.maxy - roi.miny) / 2

        if stage in ['fit']:
            train_roi = BoundingBox(
                roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt
            )
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.dataset, self.patch_size, self.batch_size, self.length, train_roi
            )
        if stage in ['fit', 'validate']:
            val_roi = BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
            self.val_sampler = GridGeoSampler(
                self.dataset, self.patch_size, self.patch_size, val_roi
            )
        if stage in ['test']:
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
