import abc
from typing import Any, Dict, Iterable

from torch.utils.data import Dataset


class GeoDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets containing geospatial information.

    Geospatial information includes things like:

    * latitude, longitude
    * time
    * coordinate reference systems (CRS)

    These kind of datasets are special because they can be combined. For example:

    * Combine Landsat8 and CDL to train a model for crop classification
    * Combine Sentinel2 and Chesapeake to train a model for land cover mapping

    This isn't true for VisionDataset, where the lack of geospatial information
    prohibits swapping image sources or target labels.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data and labels at that index
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """
        pass

    def __add__(self, other: "GeoDataset") -> "ZipDataset":  # type: ignore[override]
        """Merge two GeoDatasets.

        Parameters:
            other: another dataset

        Returns:
            a single dataset
        """
        return ZipDataset([self, other])

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: GeoDataset
    size: {len(self)}"""


class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data and labels at that index
        """
        pass

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """
        pass

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: VisionDataset
    size: {len(self)}"""


class ZipDataset(GeoDataset):
    """Dataset for merging two or more GeoDatasets.

    For example, this allows you to combine an image source like Landsat8 with a target
    label like CDL.
    """

    def __init__(self, datasets: Iterable[GeoDataset]) -> None:
        """Initialize a new Dataset instance.

        Parameters:
            datasets: list of datasets to merge
        """
        for ds in datasets:
            assert isinstance(ds, GeoDataset), "ZipDataset only supports GeoDatasets"

        self.datasets = datasets

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Parameters:
            index: index to return

        Returns:
            data and labels at that index
        """
        sample = {}
        for ds in self.datasets:
            sample.update(ds[index])
        return sample

    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """
        return min(map(len, self.datasets))

    def __add__(self, other: "GeoDataset") -> "ZipDataset":  # type: ignore[override]
        """Merge two GeoDatasets.

        Parameters:
            other: another dataset

        Returns:
            a single dataset
        """
        return ZipDataset([*self.datasets, other])

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: ZipDataset
    size: {len(self)}"""
