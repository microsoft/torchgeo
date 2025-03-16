# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench datasets."""

from typing import Any, Literal

from torch import Tensor

from ..geo import NonGeoDataset
from .base import CopernicusBenchBase
from .cloud_s2 import CopernicusBenchCloudS2
from .cloud_s3 import CopernicusBenchCloudS3

__all__ = (
    'CopernicusBench',
    'CopernicusBenchBase',
    'CopernicusBenchCloudS2',
    'CopernicusBenchCloudS3',
)

DATASET_REGISTRY = {
    'cloud_s2': CopernicusBenchCloudS2,
    'cloud_s3': CopernicusBenchCloudS3,
}


class CopernicusBench(NonGeoDataset):
    """Copernicus-Bench datasets.

    This wrapper supports dynamically loading datasets in Copernicus-Bench.

    If you use this dataset in your research, please cite the following papers:

    * TODO

    .. versionadded:: 0.7
    """

    def __init__(
        self, name: Literal['cloud_s2', 'cloud_s3'], *args: Any, **kwargs: Any
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
