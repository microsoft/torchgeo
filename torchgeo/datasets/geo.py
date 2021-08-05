"""Base classes for all :mod:`torchgeo` datasets."""

import abc
import glob
import os
import re
import sys
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import torch
from rasterio.crs import CRS
from rasterio.vrt import WarpedVRT
from rtree.index import Index, Property
from torch import Tensor
from torch.utils.data import Dataset

from .utils import BoundingBox

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Dataset.__module__ = "torch.utils.data"


class GeoDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets containing geospatial information.

    Geospatial information includes things like:

    * latitude, longitude
    * time
    * :term:`coordinate reference system (CRS)`

    These kind of datasets are special because they can be combined. For example:

    * Combine Landsat8 and CDL to train a model for crop classification
    * Combine Sentinel2 and Chesapeake to train a model for land cover mapping

    This isn't true for :class:`VisionDataset`, where the lack of geospatial information
    prohibits swapping image sources or target labels.
    """

    #: R-tree to index geospatial data. Subclasses must instantiate and insert data into
    #: this index in order for the sampler to index it properly.
    index: Index

    #: :term:`coordinate reference system (CRS)` for the dataset.
    crs: CRS

    @abc.abstractmethod
    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """

    def __add__(self, other: "GeoDataset") -> "ZipDataset":  # type: ignore[override]
        """Merge two GeoDatasets.

        Args:
            other: another dataset

        Returns:
            a single dataset

        Raises:
            ValueError: if other is not a GeoDataset, or if datasets do not overlap,
                or if datasets do not have the same
                :term:`coordinate reference system (CRS)`
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
    bbox: {self.bounds}"""

    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the index.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        return BoundingBox(*self.index.bounds)


class RasterDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset`s stored as raster files."""

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = "*"

    #: Regular expression used to extract date from filename.
    #:
    #: The expression should use named groups. The expression may contain any number of
    #: groups. The following groups are specifically searched for by the base class:
    #:
    #: * ``date``: used to calculate ``mint`` and ``maxt`` for ``index`` insertion
    #: * ``band``: used when :attr:`separate_files` is True
    filename_regex = ".*"

    #: Date format string used to parse date from filename.
    #:
    #: Not used if :attr:`filename_regex` does not contain a ``date`` group.
    date_format = "%Y%m%d"

    #: True if dataset contains imagery, False if dataset contains mask
    is_image = True

    #: True if data is stored in a separate file for each band, else False.
    separate_files = False

    #: Names of all available bands in the dataset
    all_bands: List[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: List[str] = []

    #: If True, stretch the image from the 2nd percentile to the 98th percentile,
    #: used for plotting
    stretch = False

    #: Color map for the dataset, used for plotting
    cmap: Dict[int, Tuple[int, int, int, int]] = {}

    def __init__(
        self,
        root: str,
        crs: Optional[CRS] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to project to. Uses the CRS
                of the files by default
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            FileNotFoundError: if no files are found in ``root``
        """
        self.root = root
        self.crs = crs
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        # See if file has a color map
                        try:
                            self.cmap = src.colormap(1)
                        except ValueError:
                            pass

                        if self.crs is None:
                            self.crs = src.crs

                        with WarpedVRT(src, crs=self.crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        time = datetime.strptime(date, self.date_format)
                        mint = maxt = time.timestamp()

                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(query, objects=True)

        try:
            hit = next(hits)  # TODO: this assumes there is only a single hit
        except StopIteration:
            raise IndexError(
                f"query: {query} is not within bounds of the index: {self.bounds}"
            )

        filepath = hit.object
        if self.separate_files:
            data_list = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in getattr(self, "bands", self.all_bands):
                filename = os.path.basename(filepath)
                directory = os.path.dirname(filepath)
                match = re.match(filename_regex, filename)
                if match:
                    start, end = match.start("band"), match.end("band")
                    filename = filename[:start] + band + filename[end:]
                    data_list.append(
                        self._load_file(os.path.join(directory, filename), query)
                    )
            data = torch.stack(data_list)
        else:
            data = self._load_file(filepath, query)

        key = "image" if self.is_image else "masks"
        sample = {
            key: data,
            "crs": self.crs,
            "bbox": query,
        }

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _load_file(self, filepath: str, query: BoundingBox) -> Tensor:
        """Load a single raster file.

        Args:
            filepath: path to file to open
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            image/mask at that index
        """
        with rasterio.open(filepath) as src:
            with WarpedVRT(src, crs=self.crs, nodata=0) as vrt:
                window = rasterio.windows.from_bounds(
                    query.minx,
                    query.miny,
                    query.maxx,
                    query.maxy,
                    transform=vrt.transform,
                )
                array = vrt.read(window=window).astype(np.int32)
                tensor: Tensor = torch.tensor(array)  # type: ignore[attr-defined]
                return tensor

    def plot(self, data: Tensor) -> None:
        """Plot a data sample.

        Args:
            data: the data to plot

        Raises:
            AssertionError: if ``is_image`` is True and ``data`` has a different number
                of channels than expected
        """
        array = data.squeeze().numpy()

        if self.is_image:
            bands = getattr(self, "bands", self.all_bands)
            assert array.shape[0] == len(bands)

            # Only plot RGB bands
            if bands and self.rgb_bands:
                indices = np.array([bands.index(band) for band in self.rgb_bands])
                array = array[indices]

            # Convert from CxHxW to HxWxC
            array = np.rollaxis(array, 0, 3)

        if self.cmap:
            # Convert from class labels to RGBA values
            cmap = np.array([self.cmap[i] for i in range(len(self.cmap))])
            array = cmap[array]

        if self.stretch:
            # Stretch to the range of 2nd to 98th percentile
            per02 = np.percentile(array, 2)  # type: ignore[no-untyped-call]
            per98 = np.percentile(array, 98)  # type: ignore[no-untyped-call]
            array = (array - per02) / (per98 - per02)
            array = np.clip(array, 0, 1)

        # Plot the data
        ax = plt.axes()
        ax.imshow(array)
        ax.axis("off")
        plt.show()
        plt.close()


class VectorDataset(GeoDataset):
    pass


class VisionDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and labels at that index

        Raises:
            IndexError: if index is out of range of the dataset
        """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Return the length of the dataset.

        Returns:
            length of the dataset
        """

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

    def __init__(self, datasets: Sequence[GeoDataset]) -> None:
        """Initialize a new Dataset instance.

        Args:
            datasets: list of datasets to merge

        Raises:
            ValueError: if datasets contains non-GeoDatasets, do not overlap, or are not
                in the same :term:`coordinate reference system (CRS)`
        """
        for ds in datasets:
            if not isinstance(ds, GeoDataset):
                raise ValueError("ZipDataset only supports GeoDatasets")

        crs = datasets[0].crs
        for ds in datasets:
            if ds.crs != crs:
                raise ValueError("Datasets must be in the same CRS")

        self.datasets = datasets
        self.crs = crs

        # Make sure datasets have overlap
        try:
            self.bounds
        except ValueError:
            raise ValueError("Datasets have no overlap")

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of data/labels and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query.intersects(self.bounds):
            raise IndexError(
                f"query: {query} is not within bounds of the index: {self.bounds}"
            )

        sample = {}
        for ds in self.datasets:
            sample.update(ds[query])
        return sample

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: ZipDataset
    bbox: {self.bounds}"""

    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the index.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        # We want to compute the intersection of all dataset bounds, not the union
        minx = max([ds.bounds[0] for ds in self.datasets])
        maxx = min([ds.bounds[1] for ds in self.datasets])
        miny = max([ds.bounds[2] for ds in self.datasets])
        maxy = min([ds.bounds[3] for ds in self.datasets])
        mint = max([ds.bounds[4] for ds in self.datasets])
        maxt = min([ds.bounds[5] for ds in self.datasets])

        return BoundingBox(minx, maxx, miny, maxy, mint, maxt)
