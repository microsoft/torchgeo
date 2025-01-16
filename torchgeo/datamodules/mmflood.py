# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MMFlood datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample
from torch import Tensor

from ..datasets import MMFlood
from ..samplers import GridGeoSampler, RandomBatchGeoSampler
from ..samplers.utils import _to_tuple
from .geo import GeoDataModule


class MMFloodDataModule(GeoDataModule):
    """LightningDataModule implementation for the MMFlood dataset.

    .. versionadded:: 0.7
    """

    # Computed over train set
    # VV, VH, dem, hydro
    median = torch.tensor([0.116051525, 0.025692634, 86.0, 0.0])
    std = torch.tensor([2.405442, 0.22719479, 242.74359, 0.1482505053281784])

    def __init__(
        self,
        batch_size: int = 32,
        patch_size: int | tuple[int, int] = 512,
        length: int | None = None,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        """Initialize a new MMFloodDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            patch_size: Size of each patch, either ``size`` or ``(height, width)``.
            length: Length of each training epoch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.MMFlood`.
        """
        super().__init__(
            MMFlood,
            batch_size=batch_size,
            patch_size=patch_size,
            length=length,
            num_workers=num_workers,
            **kwargs,
        )
        avg, std = self._get_mean_std(
            dem=kwargs.get('include_dem', False),
            hydro=kwargs.get('include_hydro', False),
        )

        # Using median for normalization for better stability,
        # as stated by the original authors
        self.train_aug = K.AugmentationSequential(
            K.RandomResizedCrop(_to_tuple(self.patch_size), p=0.8, scale=(0.5, 1.0)),
            K.Normalize(avg, std),
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=0.5),
            K.RandomRotation90((0, 3), p=0.5),
            K.RandomElasticTransform(sigma=(50, 50)),
            keepdim=True,
            data_keys=None,
            extra_args={
                DataKey.MASK: {'resample': Resample.NEAREST, 'align_corners': None}
            },
        )

        self.aug = K.AugmentationSequential(
            K.Normalize(avg, std), keepdim=True, data_keys=None
        )

    def _get_mean_std(
        self, dem: bool = False, hydro: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Retrieve mean and standard deviation tensors used for normalization.

        Args:
            dem: True if DEM data is loaded
            hydro: True if hydrography data is loaded

        Returns:
            mean and standard deviation tensors
        """
        idxs = [0, 1]  # VV, VH
        if dem:
            idxs.append(2)
        if hydro:
            idxs.append(3)
        return self.median[idxs], self.std[idxs]

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', 'predict'.
        """
        if stage in ['fit']:
            self.train_dataset = MMFlood(**self.kwargs, split='train')
            self.train_batch_sampler = RandomBatchGeoSampler(
                self.train_dataset, self.patch_size, self.batch_size, self.length
            )
        if stage in ['fit', 'validate']:
            self.val_dataset = MMFlood(**self.kwargs, split='val')
            self.val_sampler = GridGeoSampler(
                self.val_dataset, self.patch_size, self.patch_size
            )
        if stage in ['test']:
            self.test_dataset = MMFlood(**self.kwargs, split='test')
            self.test_sampler = GridGeoSampler(
                self.test_dataset, self.patch_size, self.patch_size
            )
