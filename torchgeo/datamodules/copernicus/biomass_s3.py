# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Biomass-S3 datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from kornia.constants import DataKey, Resample

from ...datasets import CopernicusBenchBiomassS3
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


class CopernicusBenchBiomassS3DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Copernicus Biomass-S3 dataset.

    Uses the train/val/test splits provided with the benchmark.

    .. versionadded:: 0.81
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
        bands = kwargs.get('bands', SCALE.keys())
        mode = kwargs.get('mode', 'static')
        scale_factors = torch.tensor([SCALE[b] for b in bands], dtype=torch.float32)

        self.mean = torch.zeros(len(bands), dtype=torch.float32)
        self.std = torch.reciprocal(scale_factors)

        resize_transform = K.AugmentationSequential(
            K.Resize(
                size=TARGET_SIZE, resample=Resample.BILINEAR.name, align_corners=False
            ),
            data_keys=None,
            keepdim=True,
            extra_args={
                DataKey.MASK: {'resample': Resample.NEAREST, 'align_corners': None}
            },
        )
        existing_transform = kwargs.get('transforms')

        if existing_transform is not None:

            def composed(sample: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
                sample = resize_transform(sample)
                return existing_transform(sample)

            kwargs['transforms'] = composed
        else:
            kwargs['transforms'] = resize_transform

        super().__init__(CopernicusBenchBiomassS3, batch_size, num_workers, **kwargs)

        normalizer = K.Normalize(mean=self.mean, std=self.std)
        if mode == 'time-series':
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
