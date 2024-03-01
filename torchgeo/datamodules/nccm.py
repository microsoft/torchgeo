# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Northeastern China Crop Map (NCCM) datamodule."""

from typing import Any, Optional, Union

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from matplotlib.figure import Figure

from ..datasets import NCCM, Sentinel2, random_grid_cell_assignment
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class NCCMSentinel2DataModule(GeoDataModule):
    """LightningDataModule implementation for the NCCM and Sentinel2 datasets.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.6
    """

    def __init__(
        self,
        batch_size: int = 64,
        patch_size: Union[int, tuple[int, int]] = 16,
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
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(self.patch_size),
            data_keys=["image", "mask"],
        )

    def setup(self, stage: str) -> None:
        """Set up datasets and samplers.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.sentinel2 = Sentinel2(**self.sentinel2_kwargs)
        self.nccm = NCCM(**self.nccm_kwargs)
        generator = torch.Generator().manual_seed(0)

        self.dataset = self.sentinel2 & self.nccm
        (self.train_dataset, self.val_dataset, self.test_dataset) = (
            random_grid_cell_assignment(self.dataset, [0.8, 0.1, 0.1], 10, generator)
        )

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
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
        """Run NCCM plot method.

        Args:
            *args: Arguments passed to plot method.
            **kwargs: Keyword arguments passed to plot method.

        Returns:
            A matplotlib Figure with the image, ground truth, and predictions.
        """
        return self.nccm.plot(*args, **kwargs)
