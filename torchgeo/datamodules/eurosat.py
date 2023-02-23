# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT datamodule."""

from typing import Any

import torch

from ..datasets import EuroSAT, EuroSAT100
from .geo import NonGeoDataModule

MEAN = torch.tensor(
    [
        1354.40546513,
        1118.24399958,
        1042.92983953,
        947.62620298,
        1199.47283961,
        1999.79090914,
        2369.22292565,
        2296.82608323,
        732.08340178,
        12.11327804,
        1819.01027855,
        1118.92391149,
        2594.14080798,
    ]
)

STD = torch.tensor(
    [
        245.71762908,
        333.00778264,
        395.09249139,
        593.75055589,
        566.4170017,
        861.18399006,
        1086.63139075,
        1117.98170791,
        404.91978886,
        4.77584468,
        1002.58768311,
        761.30323499,
        1231.58581042,
    ]
)


class EuroSATDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EuroSAT dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

    mean = MEAN
    std = STD

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new EuroSATDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.EuroSAT`.
        """
        super().__init__(EuroSAT, batch_size, num_workers, **kwargs)


class EuroSAT100DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EuroSAT100 dataset.

    Intended for tutorials and demonstrations, not for benchmarking.

    .. versionadded:: 0.5
    """

    mean = MEAN
    std = STD

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new EuroSAT100DataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.EuroSAT100`.
        """
        super().__init__(EuroSAT100, batch_size, num_workers, **kwargs)
