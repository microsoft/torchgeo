# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Seasonal Contrast datamodule."""

from typing import Any

import torch

from ..datasets import SeasonalContrastS2
from .geo import NonGeoDataModule


class SeasonalContrastS2DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the Seasonal Contrast dataset.

    .. versionadded:: 0.5
    """

    # https://github.com/ServiceNow/seasonal-contrast/blob/8285173ec205b64bc3e53b880344dd6c3f79fa7a/datasets/bigearthnet_dataset.py#L13  # noqa: E501
    mean = torch.tensor(
        [
            340.76769064,
            429.9430203,
            614.21682446,
            590.23569706,
            950.68368468,
            1792.46290469,
            2075.46795189,
            2218.94553375,
            2266.46036911,
            2246.0605464,
            1594.42694882,
            1009.32729131,
        ]
    )
    std = 2 * torch.tensor(
        [
            554.81258967,
            572.41639287,
            582.87945694,
            675.88746967,
            729.89827633,
            1096.01480586,
            1273.45393088,
            1365.45589904,
            1356.13789355,
            1302.3292881,
            1079.19066363,
            818.86747235,
        ]
    )

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
        bands = kwargs.get("bands", SeasonalContrastS2.rgb_bands)
        all_bands = SeasonalContrastS2.all_bands
        indices = [all_bands.index(band) for band in bands]
        self.mean = self.mean[indices]
        self.std = self.std[indices]

        super().__init__(SeasonalContrastS2, batch_size, num_workers, **kwargs)

    def setup(self, stage: str) -> None:
        """Set up datasets.

        Args:
            stage: Either 'fit', 'validate', 'test', or 'predict'.
        """
        self.dataset = SeasonalContrastS2(**self.kwargs)
