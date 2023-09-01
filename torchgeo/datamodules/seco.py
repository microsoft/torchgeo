# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Seasonal Contrast datamodule."""

from typing import Any

import kornia.augmentation as K
import torch
from einops import repeat

from ..datasets import SeasonalContrastS2
from ..transforms import AugmentationSequential
from .geo import NonGeoDataModule


class SeasonalContrastS2DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Seasonal Contrast dataset.

    .. versionadded:: 0.5
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new SeasonalContrastS2DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.SeasonalContrastS2`.
        """
        super().__init__(SeasonalContrastS2, batch_size, num_workers, **kwargs)

        bands = kwargs.get("bands", SeasonalContrastS2.rgb_bands)
        seasons = kwargs.get("seasons", 1)

        # Normalization only available for RGB dataset, defined here:
        # https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/seco_dataset.py  # noqa: E501
        if bands == SeasonalContrastS2.rgb_bands:
            _min = torch.tensor([3, 2, 0])
            _max = torch.tensor([88, 103, 129])
            _mean = torch.tensor([0.485, 0.456, 0.406])
            _std = torch.tensor([0.229, 0.224, 0.225])

            _min = repeat(_min, "c -> (t c)", t=seasons)
            _max = repeat(_max, "c -> (t c)", t=seasons)
            _mean = repeat(_mean, "c -> (t c)", t=seasons)
            _std = repeat(_std, "c -> (t c)", t=seasons)

            self.aug = AugmentationSequential(
                K.Normalize(mean=_min, std=_max - _min),
                K.Normalize(mean=torch.tensor(0), std=1 / torch.tensor(255)),
                K.Normalize(mean=_mean, std=_std),
                data_keys=["image"],
            )

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SeasonalContrastS2(**self.kwargs)
