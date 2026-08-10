# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Biomass-S3 datamodule."""

from collections.abc import Callable
from functools import partial
from typing import Any, cast

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample

from ...datasets import CopernicusBenchBiomassS3
from ...datasets.utils import pad_across_batches
from ..geo import NonGeoDataModule

# Multiplicative scale factors from
# https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S3_OLCI#bands
SCALE = {
    'Oa01_radiance': 0.0139465,
    'Oa02_radiance': 0.0133873,
    'Oa03_radiance': 0.0121481,
    'Oa04_radiance': 0.0115198,
    'Oa05_radiance': 0.0100953,
    'Oa06_radiance': 0.0123538,
    'Oa07_radiance': 0.00879161,
    'Oa08_radiance': 0.00876539,
    'Oa09_radiance': 0.0095103,
    'Oa10_radiance': 0.00773378,
    'Oa11_radiance': 0.00675523,
    'Oa12_radiance': 0.0071996,
    'Oa13_radiance': 0.00749684,
    'Oa14_radiance': 0.0086512,
    'Oa15_radiance': 0.00526779,
    'Oa16_radiance': 0.00530267,
    'Oa17_radiance': 0.00493004,
    'Oa18_radiance': 0.00549962,
    'Oa19_radiance': 0.00502847,
    'Oa20_radiance': 0.00326378,
    'Oa21_radiance': 0.00324118,
}

TARGET_SIZE = (282, 282)
# Dataset-wide biomass statistics after resizing masks to TARGET_SIZE.
TARGET_MEAN = 93.197777079690
TARGET_STD = 119.004185235754


class CopernicusBenchBiomassS3DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Copernicus Biomass-S3 dataset.

    Uses the train/val/test splits provided with the benchmark.

    .. versionadded:: 0.10
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new CopernicusBenchBiomassS3DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.CopernicusBenchBiomassS3`.
        """
        bands = kwargs.get('bands', CopernicusBenchBiomassS3.all_bands)
        mode = kwargs.get('mode', 'static')
        scale_factors = torch.tensor([SCALE[b] for b in bands], dtype=torch.float32)

        self.mean = torch.zeros(len(bands), dtype=torch.float32)
        self.std = torch.reciprocal(scale_factors)
        self.target_mean = TARGET_MEAN
        self.target_std = TARGET_STD

        resize_transform = K.AugmentationSequential(
            K.Resize(size=TARGET_SIZE, resample=Resample.BILINEAR, align_corners=False),
            data_keys=None,
            keepdim=True,
            extra_args={
                DataKey.MASK: {'resample': Resample.BILINEAR, 'align_corners': None}
            },
        )
        existing_transform = cast(
            Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]] | None,
            kwargs.get('transforms'),
        )

        if existing_transform is not None:
            transform = existing_transform

            def composed(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                resized = resize_transform(sample)
                return transform(resized)

            kwargs['transforms'] = composed
        else:
            kwargs['transforms'] = resize_transform

        super().__init__(CopernicusBenchBiomassS3, batch_size, num_workers, **kwargs)

        normalizer = K.Normalize(mean=self.mean, std=self.std)
        if mode == 'time-series':
            self.collate_fn = partial(pad_across_batches, padding_length=4)
            self.aug = K.AugmentationSequential(
                K.VideoSequential(normalizer),
                data_keys=None,
                keepdim=True,
                same_on_batch=True,
            )
        else:
            self.aug = K.AugmentationSequential(
                normalizer, data_keys=None, keepdim=True
            )

    def on_after_batch_transfer(
        self, batch: dict[str, torch.Tensor], dataloader_idx: int
    ) -> dict[str, torch.Tensor]:
        """Normalize imagery and biomass targets.

        Args:
            batch: A batch of data that needs to be altered or augmented.
            dataloader_idx: The index of the dataloader to which the batch belongs.

        Returns:
            A batch with normalized imagery and target masks.
        """
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        if 'mask' in batch:
            batch['mask'] = (batch['mask'] - self.target_mean) / self.target_std
        return batch
