# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""National Agriculture Imagery Program (NAIP) datamodule."""

from typing import Any, Tuple, Union

import matplotlib.pyplot as plt

from ..datasets import NAIP, BoundingBox, Chesapeake13
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from .geo import GeoDataModule


class NAIPChesapeakeDataModule(GeoDataModule):
    """LightningDataModule implementation for the NAIP and Chesapeake datasets.

    Uses the train/val/test splits from the dataset.
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, Tuple[int, int]] = 256,
        length: int = 1000,
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
                (prefix keys with ``chesapeake_``)
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

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        chesapeake = Chesapeake13(**self.chesapeake_kwargs)
        naip = NAIP(**self.naip_kwargs)
        self.dataset = chesapeake & naip

        roi = self.dataset.bounds
        midx = roi.minx + (roi.maxx - roi.minx) / 2
        midy = roi.miny + (roi.maxy - roi.miny) / 2

        if stage in ["fit"]:
            roi = BoundingBox(roi.minx, midx, roi.miny, roi.maxy, roi.mint, roi.maxt)
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.dataset, self.patch_size, self.batch_size, self.length, roi
            )
        if stage in ["fit", "validate"]:
            roi = BoundingBox(midx, roi.maxx, roi.miny, midy, roi.mint, roi.maxt)
            self.val_sampler = GridGeoSampler(
                self.dataset, self.patch_size, self.patch_size, roi
            )
        if stage in ["test"]:
            roi = BoundingBox(roi.minx, roi.maxx, midy, roi.maxy, roi.mint, roi.maxt)
            self.test_sampler = GridGeoSampler(
                self.dataset, self.patch_size, self.patch_size, roi
            )

    def plot(self, *args: Any, **kwargs: Any) -> Tuple[plt.Figure, plt.Figure]:
        """Run NAIP and Chesapeake plot methods.

        See :meth:`torchgeo.datasets.NAIP.plot` and
        :meth:`torchgeo.datasets.Chesapeake.plot`.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A list of matplotlib Figures with the image, ground truth, and predictions.

        .. versionadded:: 0.4
        """
        image = self.naip.plot(*args, **kwargs)
        label = self.chesapeake.plot(*args, **kwargs)
        return image, label
