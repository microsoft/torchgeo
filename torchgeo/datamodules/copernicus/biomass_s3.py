# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench Biomass-S3 datamodule."""

from typing import Any

import kornia.augmentation as K
import torch

from ...datasets import CopernicusBenchBiomassS3
from ..geo import NonGeoDataModule


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
        bands = kwargs.get('bands', CopernicusBenchBiomassS3.all_bands)
        mode = kwargs.get('mode', 'static')

        self.mean = torch.zeros(len(bands), dtype=torch.float32)
        self.std = torch.ones(len(bands), dtype=torch.float32)

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
