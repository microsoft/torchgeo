# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench datasets."""

from typing import Any, Literal

from torch import Tensor

from ..geo import NonGeoDataset
from .base import CopernicusBenchBase
from .bigearthnet_s1 import CopernicusBenchBigEarthNetS1
from .bigearthnet_s2 import CopernicusBenchBigEarthNetS2
from .biomass_s3 import CopernicusBenchBiomassS3
from .cloud_s2 import CopernicusBenchCloudS2
from .cloud_s3 import CopernicusBenchCloudS3
from .dfc2020_s1 import CopernicusBenchDFC2020S1
from .dfc2020_s2 import CopernicusBenchDFC2020S2
from .eurosat_s1 import CopernicusBenchEuroSATS1
from .eurosat_s2 import CopernicusBenchEuroSATS2
from .flood_s1 import CopernicusBenchFloodS1
from .lc100cls_s3 import CopernicusBenchLC100ClsS3
from .lc100seg_s3 import CopernicusBenchLC100SegS3
from .lcz_s2 import CopernicusBenchLCZS2

__all__ = (
    'CopernicusBench',
    'CopernicusBenchBase',
    'CopernicusBenchBigEarthNetS1',
    'CopernicusBenchBigEarthNetS2',
    'CopernicusBenchBiomassS3',
    'CopernicusBenchCloudS2',
    'CopernicusBenchCloudS3',
    'CopernicusBenchDFC2020S1',
    'CopernicusBenchDFC2020S2',
    'CopernicusBenchEuroSATS1',
    'CopernicusBenchEuroSATS2',
    'CopernicusBenchFloodS1',
    'CopernicusBenchLC100ClsS3',
    'CopernicusBenchLC100SegS3',
    'CopernicusBenchLCZS2',
)

DATASET_REGISTRY = {
    'cloud_s2': CopernicusBenchCloudS2,
    'cloud_s3': CopernicusBenchCloudS3,
    'eurosat_s1': CopernicusBenchEuroSATS1,
    'eurosat_s2': CopernicusBenchEuroSATS2,
    'bigearthnet_s1': CopernicusBenchBigEarthNetS1,
    'bigearthnet_s2': CopernicusBenchBigEarthNetS2,
    'lc100cls_s3': CopernicusBenchLC100ClsS3,
    'lc100seg_s3': CopernicusBenchLC100SegS3,
    'dfc2020_s1': CopernicusBenchDFC2020S1,
    'dfc2020_s2': CopernicusBenchDFC2020S2,
    'flood_s1': CopernicusBenchFloodS1,
    'lcz_s2': CopernicusBenchLCZS2,
    'biomass_s3': CopernicusBenchBiomassS3,
}


class CopernicusBench(NonGeoDataset):
    """Copernicus-Bench datasets.

    This wrapper supports dynamically loading datasets in Copernicus-Bench.

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2503.11849

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        name: Literal[
            'cloud_s2',
            'cloud_s3',
            'eurosat_s1',
            'eurosat_s2',
            'bigearthnet_s1',
            'bigearthnet_s2',
            'lc100cls_s3',
            'lc100seg_s3',
            'dfc2020_s1',
            'dfc2020_s2',
            'flood_s1',
            'lcz_s2',
            'biomass_s3',
        ],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Initialize a new CopernicusBench instance.

        Args:
            name: Name of the dataset to load.
            *args: Arguments to pass to dataset class.
            **kwargs: Keyword arguments to pass to dataset class.
        """
        self.name = name
        self.dataset: CopernicusBenchBase = DATASET_REGISTRY[name](*args, **kwargs)

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            Length of the dataset.
        """
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: Index to return.

        Returns:
            Data and labels at that index.
        """
        return self.dataset[index]

    def __getattr__(self, name: str) -> Any:
        """Wrapper around dataset object."""
        return getattr(self.dataset, name)
