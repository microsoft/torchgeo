# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Copernicus-Bench datasets."""

from typing import Any, Literal

from torch import Tensor

from ..geo import NonGeoDataset
from .base import CopernicusBenchBase
from .cloud_s2 import CopernicusBenchCloudS2

DATASET_REGISTRY = {'cloud_s2': CopernicusBenchCloudS2}


class CopernicusBench(NonGeoDataset):
    """Copernicus-Bench datasets.

    This wrapper supports dynamically loading datasets in Copernicus-Bench.

    If you use this dataset in your research, please cite the following papers:

    * TODO

    .. versionadded:: 0.7
    """

    def __init__(self, dataset: Literal['cloud_s2'], *args: Any, **kwargs: Any) -> None:
        """Initialize a new CopernicusBench instance.

        Args:
            dataset: Name of the dataset to load.
            *args: Arguments to pass to dataset class.
            **kwargs: Keyword arguments to pass to dataset class.
        """
        self.dataset: CopernicusBenchBase = DATASET_REGISTRY[dataset](*args, **kwargs)

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
