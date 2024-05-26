# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT datamodule."""

from typing import Any

import torch

from ..datasets import EuroSAT, EuroSAT100, EuroSATSpatial
from .geo import NonGeoDataModule

MEAN = {
    'B01': 1354.40546513,
    'B02': 1118.24399958,
    'B03': 1042.92983953,
    'B04': 947.62620298,
    'B05': 1199.47283961,
    'B06': 1999.79090914,
    'B07': 2369.22292565,
    'B08': 2296.82608323,
    'B8A': 732.08340178,
    'B09': 12.11327804,
    'B10': 1819.01027855,
    'B11': 1118.92391149,
    'B12': 2594.14080798,
}

STD = {
    'B01': 245.71762908,
    'B02': 333.00778264,
    'B03': 395.09249139,
    'B04': 593.75055589,
    'B05': 566.4170017,
    'B06': 861.18399006,
    'B07': 1086.63139075,
    'B08': 1117.98170791,
    'B8A': 404.91978886,
    'B09': 4.77584468,
    'B10': 1002.58768311,
    'B11': 761.30323499,
    'B12': 1231.58581042,
}


class EuroSATDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EuroSAT dataset.

    Uses the train/val/test splits from the dataset.

    .. versionadded:: 0.2
    """

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

        bands = kwargs.get('bands', EuroSAT.all_band_names)
        self.mean = torch.tensor([MEAN[b] for b in bands])
        self.std = torch.tensor([STD[b] for b in bands])


class EuroSAT100DataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EuroSAT100 dataset.

    Intended for tutorials and demonstrations, not for benchmarking.

    .. versionadded:: 0.5
    """

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

        bands = kwargs.get('bands', EuroSAT.all_band_names)
        self.mean = torch.tensor([MEAN[b] for b in bands])
        self.std = torch.tensor([STD[b] for b in bands])


class EuroSATSpatialDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EuroSATSpatial dataset.

    Intended for tutorials and demonstrations, not for benchmarking.

    .. versionadded:: 0.5
    """

    def __init__(
        self, batch_size: int = 64, num_workers: int = 0, **kwargs: Any
    ) -> None:
        """Initialize a new EuroSATSpatialDataModule instance.

        Args:
            batch_size: Size of each mini-batch.
            num_workers: Number of workers for parallel data loading.
            **kwargs: Additional keyword arguments passed to
                :class:`~torchgeo.datasets.EuroSATSpatial`.
        """
        super().__init__(EuroSATSpatial, batch_size, num_workers, **kwargs)

        bands = kwargs.get('bands', EuroSAT.all_band_names)
        self.mean = torch.tensor([MEAN[b] for b in bands])
        self.std = torch.tensor([STD[b] for b in bands])
