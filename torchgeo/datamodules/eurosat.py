# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EuroSAT datamodule."""

from typing import Any

import torch

from ..datasets import EuroSAT, EuroSAT100, EuroSATSpatial
from .geo import NonGeoDataModule

SPATIAL_MEAN = {
    'B01': 1375.9932,
    'B02': 1142.6339,
    'B03': 1077.5502,
    'B04': 1003.8445,
    'B05': 1280.7300,
    'B06': 2130.3491,
    'B07': 2524.0549,
    'B08': 2454.1938,
    'B8A': 785.4963,
    'B09': 12.4639,
    'B10': 1969.9224,
    'B11': 1206.2421,
    'B12': 2779.4104,
}

SPATIAL_STD = {
    'B01': 249.8516,
    'B02': 337.9465,
    'B03': 392.5661,
    'B04': 612.4237,
    'B05': 562.2878,
    'B06': 806.8271,
    'B07': 1022.6378,
    'B08': 1065.4312,
    'B8A': 410.5831,
    'B09': 4.8878,
    'B10': 958.4751,
    'B11': 740.6196,
    'B12': 1157.2896,
}

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
        bands = kwargs.get('bands', EuroSAT.all_band_names)
        self.mean = torch.tensor([MEAN[b] for b in bands])
        self.std = torch.tensor([STD[b] for b in bands])
        super().__init__(EuroSAT, batch_size, num_workers, **kwargs)


class EuroSATSpatialDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the EuroSATSpatial dataset.

    Uses the spatial train/val/test splits from the dataset.

    .. versionadded:: 0.6
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
        bands = kwargs.get('bands', EuroSAT.all_band_names)
        self.mean = torch.tensor([SPATIAL_MEAN[b] for b in bands])
        self.std = torch.tensor([SPATIAL_STD[b] for b in bands])
        super().__init__(EuroSATSpatial, batch_size, num_workers, **kwargs)


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
        bands = kwargs.get('bands', EuroSAT.all_band_names)
        self.mean = torch.tensor([MEAN[b] for b in bands])
        self.std = torch.tensor([STD[b] for b in bands])
        super().__init__(EuroSAT100, batch_size, num_workers, **kwargs)
