# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""CDLSentinel2 datamodule."""

from typing import Any, Optional, Union

import torch
import kornia.augmentation as K
from kornia.constants import DataKey, Resample

from ..datasets import CDL, Sentinel2, random_grid_cell_assignment
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..samplers.utils import _to_tuple
from ..transforms import AugmentationSequential
from .geo import GeoDataModule


class CDLSentinel2DataModule(GeoDataModule):
    """LightningDataModule implementation for the CDL dataset.

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
        """Initialize a new CDLSentinel2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CDL`.
        """
        self.cdl_kwargs = {}
        self.sentinel2_kwargs = {}
        for key, val in kwargs.items():
            if key.startswith("cdl_"):
                self.cdl_kwargs[key[5:]] = val
            elif key.startswith("sentinel2_"):
                self.sentinel2_kwargs[key[10:]] = val

        super().__init__(
            CDL, batch_size, patch_size, length, num_workers, **self.cdl_kwargs
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

        self.val_aug = AugmentationSequential(
            K.Normalize(mean=self.mean, std=self.std),
            K.CenterCrop(self.patch_size),
            data_keys=["image", "mask"],
        )
        self.test_aug = AugmentationSequential(
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
        self.cdl = CDL(**self.cdl_kwargs)
        self.dataset = self.sentinel2 & self.cdl

        generator = torch.Generator().manual_seed(0)

        (self.train_dataset, self.val_dataset, self.test_dataset) = (
            random_grid_cell_assignment(self.dataset, [0.8, 0.1, 0.1], generator)
        )

        if stage in ["fit"]:
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )

        if stage in ["fit", "validate"]:
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )