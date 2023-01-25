# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all :mod:`torchgeo` datasets."""

import abc
import datetime
import functools
import glob
import os
import re
import sys
import warnings
from collections import defaultdict
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import fiona
import fiona.transform
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import rasterio
import rasterio.merge
import shapely
import torch
from matplotlib.widgets import Slider
from rasterio.crs import CRS
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds
from rtree.index import Index, Property
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader

from .utils import BoundingBox, concat_samples, disambiguate_timestamp, merge_samples


class GeoDataset(Dataset[Dict[str, Any]], abc.ABC):
    """Abstract base class for datasets containing geospatial information.

    Geospatial information includes things like:

    * coordinates (latitude, longitude)
    * :term:`coordinate reference system (CRS)`
    * resolution

    :class:`GeoDataset` is a special class of datasets. Unlike :class:`NonGeoDataset`,
    the presence of geospatial information allows two or more datasets to be combined
    based on latitude/longitude. This allows users to do things like:

    * Combine image and target labels and sample from both simultaneously
      (e.g. Landsat and CDL)
    * Combine datasets for multiple image sources for multimodal learning or data fusion
      (e.g. Landsat and Sentinel)

    These combinations require that all queries are present in *both* datasets,
    and can be combined using an :class:`IntersectionDataset`:

    .. code-block:: python

       dataset = landsat & cdl

    Users may also want to:

    * Combine datasets for multiple image sources and treat them as equivalent
      (e.g. Landsat 7 and Landsat 8)
    * Combine datasets for disparate geospatial locations
      (e.g. Chesapeake NY and PA)

    These combinations require that all queries are present in *at least one* dataset,
    and can be combined using a :class:`UnionDataset`:

    .. code-block:: python

       dataset = landsat7 | landsat8
    """

    #: Resolution of the dataset in units of CRS.
    res: float
    _crs: CRS

    # NOTE: according to the Python docs:
    #
    # * https://docs.python.org/3/library/exceptions.html#NotImplementedError
    #
    # the correct way to handle __add__ not being supported is to set it to None,
    # not to return NotImplemented or raise NotImplementedError. The downside of
    # this is that we have no way to explain to a user why they get an error and
    # what they should do instead (use __and__ or __or__).

    #: :class:`GeoDataset` addition can be ambiguous and is no longer supported.
    #: Users should instead use the intersection or union operator.
    __add__ = None  # type: ignore[assignment]

    def __init__(
        self, transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        """
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

    @abc.abstractmethod
    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """

    def __and__(self, other: "GeoDataset") -> "IntersectionDataset":
        """Take the intersection of two :class:`GeoDataset`.

        Args:
            other: another dataset

        Returns:
            a single dataset

        Raises:
            ValueError: if other is not a :class:`GeoDataset`

        .. versionadded:: 0.2
        """
        return IntersectionDataset(self, other)

    def __or__(self, other: "GeoDataset") -> "UnionDataset":
        """Take the union of two GeoDatasets.

        Args:
            other: another dataset

        Returns:
            a single dataset

        Raises:
            ValueError: if other is not a :class:`GeoDataset`

        .. versionadded:: 0.2
        """
        return UnionDataset(self, other)

    def __len__(self) -> int:
        """Return the number of files in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.index)

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: GeoDataset
    bbox: {self.bounds}
    size: {len(self)}"""

    # NOTE: This hack should be removed once the following issue is fixed:
    # https://github.com/Toblerity/rtree/issues/87

    def __getstate__(
        self,
    ) -> Tuple[Dict[str, Any], List[Tuple[Any, Any, Optional[Any]]]]:
        """Define how instances are pickled.

        Returns:
            the state necessary to unpickle the instance
        """
        objects = self.index.intersection(self.index.bounds, objects=True)
        tuples = [(item.id, item.bounds, item.object) for item in objects]
        return self.__dict__, tuples

    def __setstate__(
        self,
        state: Tuple[
            Dict[Any, Any],
            List[Tuple[int, Tuple[float, float, float, float, float, float], str]],
        ],
    ) -> None:
        """Define how to unpickle an instance.

        Args:
            state: the state of the instance when it was pickled
        """
        attrs, tuples = state
        self.__dict__.update(attrs)
        for item in tuples:
            self.index.insert(*item)

    @property
    def bounds(self) -> BoundingBox:
        """Bounds of the index.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) of the dataset
        """
        return BoundingBox(*self.index.bounds)

    @property
    def crs(self) -> CRS:
        """:term:`coordinate reference system (CRS)` for the dataset.

        Returns:
            the :term:`coordinate reference system (CRS)`

        .. versionadded:: 0.2
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of a GeoDataset.

        If ``new_crs == self.crs``, does nothing, otherwise updates the R-tree index.

        Args:
            new_crs: new :term:`coordinate reference system (CRS)`

        .. versionadded:: 0.2
        """
        if new_crs == self._crs:
            return

        new_index = Index(interleaved=False, properties=Property(dimension=3))

        project = pyproj.Transformer.from_crs(
            pyproj.CRS(str(self._crs)), pyproj.CRS(str(new_crs)), always_xy=True
        ).transform
        for hit in self.index.intersection(self.index.bounds, objects=True):
            old_minx, old_maxx, old_miny, old_maxy, mint, maxt = hit.bounds
            old_box = shapely.geometry.box(old_minx, old_miny, old_maxx, old_maxy)
            new_box = shapely.ops.transform(project, old_box)
            new_minx, new_miny, new_maxx, new_maxy = new_box.bounds
            new_bounds = (new_minx, new_maxx, new_miny, new_maxy, mint, maxt)
            new_index.insert(hit.id, new_bounds, hit.object)

        self._crs = new_crs
        self.index = new_index


class RasterDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files."""

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
    #:
    #: When :attr:`~RasterDataset.separate_files` is True, the following additional
    #: groups are searched for to find other files:
    #:
    #: * ``band``: replaced with requested band name
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

    #: Color map for the dataset, used for plotting
    cmap: Dict[int, Tuple[int, int, int, int]] = {}

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: Optional[float] = None,
        bands: Optional[Sequence[str]] = None,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        cache: bool = True,
        as_time_series: bool = False,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling
            as_time_series: whether or not to return sampled query as a time
                series or as a merged single time instance

        Raises:
            FileNotFoundError: if no files are found in ``root``

        .. versionchanged:: 0.4
            Add *as_time_series* parameter to support time series datasets
        """
        super().__init__(transforms)

        self.root = root
        self.cache = cache
        self.as_time_series = as_time_series

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs
                        if res is None:
                            res = src.res[0]

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if "date" in match.groupdict():
                        date = match.group("date")
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        if bands and self.all_bands:
            band_indexes = [self.all_bands.index(i) + 1 for i in bands]
            self.bands = bands
            assert len(band_indexes) == len(self.bands)
        elif bands:
            msg = (
                f"{self.__class__.__name__} is missing an `all_bands` attribute,"
                " so `bands` cannot be specified."
            )
            raise AssertionError(msg)
        else:
            band_indexes = None
            self.bands = self.all_bands

        self.band_indexes = band_indexes
        self._crs = cast(CRS, crs)
        self.res = cast(float, res)

    def __getitem__(
        self, query: BoundingBox
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: one or more (minx, maxx, miny, maxy, mint, maxt)
                coordinates to index

        Returns:
            sample of image/mask and metadata for each index in the query or

        Raises:
            IndexError: if queries is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(List[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        if self.separate_files:
            data_list: List[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if "date" in match.groupdict():
                            start = match.start("band")
                            end = match.end("band")
                            filename = filename[:start] + band + filename[end:]
                    filepath = glob.glob(os.path.join(directory, filename))[0]
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            if self.as_time_series:  # for timeseries seq_len x ch x height x width
                data = torch.cat(data_list, dim=1)
            else:  # ch x height x width
                data = torch.cat(data_list, dim=0)
        else:
            data = self._merge_files(filepaths, query, self.band_indexes)

        dates = list({int(f.split("/")[-1].split("_")[0]) for f in filepaths})
        dates = sorted(dates)
        key = "image" if self.is_image else "mask"
        sample = {key: data, "crs": self.crs, "bbox": query, "dates": dates}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _dataset_reader_to_array(
        self,
        vrts: List[rasterio.io.DatasetReader],
        query: BoundingBox,
        bounds: Tuple[float],
        band_indexes: Optional[Sequence[int]] = None,
    ) -> np.ndarray:
        """Read dataset readers into numpy arrays.

        Args:
            vrts: List of opened rasterio filehandles

        Returns:
            data array from file handle

        .. versionadded:: 0.4
        """
        if len(vrts) == 1:
            dest = self._read_single_file(vrts[0], band_indexes, query, bounds)
        else:
            dest, _ = rasterio.merge.merge(vrts, bounds, self.res, indexes=band_indexes)
        return dest

    def _merge_files(
        self,
        filepaths: Sequence[str],
        query: BoundingBox,
        band_indexes: Optional[Sequence[int]] = None,
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            band_indexes: indexes of bands to be used

        Returns:
            image/mask at that index

        .. versionchanged:: 0.4
            Given the *as_time_series* parameter merge files to a single time instance
            as previously or return a time series for the geographical location
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        if self.as_time_series:
            # group images with the same time-stamp for the same geographic area
            # and use rasterio merge to extract one common area
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            date_tile_dict: DefaultDict[
                str, List[rasterio.io.DatasetReader]
            ] = defaultdict(list)
            for vrt in vrt_fhs:
                match = re.match(filename_regex, os.path.basename(vrt.name))
                if match:
                    date_tile_dict[match.group("date")].append(vrt)

            date_array_dict: Dict[str, np.ndarray] = {}

            for date, vrts in date_tile_dict.items():
                date_array_dict[date] = self._dataset_reader_to_array(
                    vrts, query, bounds, band_indexes
                )

            # order the time-stamps correctly to build a sequential time series
            sorted_dates = sorted(
                list(date_array_dict.keys()),
                key=lambda x: datetime.datetime.strptime(x, self.date_format),
            )
            # subsequently stack these extracted patches along the timedimension
            # specify bounds and according to these bounds read out
            dest = np.stack([date_array_dict[date] for date in sorted_dates])
        else:
            dest = self._dataset_reader_to_array(vrt_fhs, query, bounds, band_indexes)

        # fix numpy dtypes which are not supported by pytorch tensors
        if dest.dtype == np.uint16:
            dest = dest.astype(np.int32)
        elif dest.dtype == np.uint32:
            dest = dest.astype(np.int64)

        tensor = torch.tensor(dest)  # dimension seq_len x 1 x height x width
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: str) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: str) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
            src.close()
            return vrt
        else:
            return src

    def _read_single_file(
        self,
        src: rasterio.io.DatasetReader,
        band_indexes: Optional[Sequence[int]],
        query: BoundingBox,
        bounds: Tuple[float, float, float, float],
    ) -> np.ndarray:
        """Read a single datasetreader from a query into array."""
        out_width = round((query.maxx - query.minx) / self.res)
        out_height = round((query.maxy - query.miny) / self.res)
        count = len(band_indexes) if band_indexes else src.count
        out_shape = (count, out_height, out_width)
        dest = src.read(
            indexes=band_indexes,
            out_shape=out_shape,
            window=from_bounds(*bounds, src.transform),
        )
        return dest


class VectorDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as vector files."""

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = "*"

    def __init__(
        self,
        root: str = "data",
        crs: Optional[CRS] = None,
        res: float = 0.0001,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        label_name: Optional[str] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            root: root directory where dataset can be found
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            label_name: name of the dataset property that has the label to be
                rasterized into the mask

        Raises:
            FileNotFoundError: if no files are found in ``root``

        .. versionadded:: 0.4
            The *label_name* parameter.
        """
        super().__init__(transforms)

        self.root = root
        self.res = res
        self.label_name = label_name

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, "**", self.filename_glob)
        for filepath in glob.iglob(pathname, recursive=True):
            try:
                with fiona.open(filepath) as src:
                    if crs is None:
                        crs = CRS.from_dict(src.crs)

                    minx, miny, maxx, maxy = src.bounds
                    (minx, maxx), (miny, maxy) = fiona.transform.transform(
                        src.crs, crs.to_dict(), [minx, maxx], [miny, maxy]
                    )
            except fiona.errors.FionaValueError:
                # Skip files that fiona is unable to read
                continue
            else:
                mint = 0
                maxt = sys.maxsize
                coords = (minx, maxx, miny, maxy, mint, maxt)
                self.index.insert(i, coords, filepath)
                i += 1

        if i == 0:
            raise FileNotFoundError(
                f"No {self.__class__.__name__} data was found in '{root}'"
            )

        self._crs = crs

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = [hit.object for hit in hits]

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        shapes = []
        for filepath in filepaths:
            with fiona.open(filepath) as src:
                # We need to know the bounding box of the query in the source CRS
                (minx, maxx), (miny, maxy) = fiona.transform.transform(
                    self.crs.to_dict(),
                    src.crs,
                    [query.minx, query.maxx],
                    [query.miny, query.maxy],
                )

                # Filter geometries to those that intersect with the bounding box
                for feature in src.filter(bbox=(minx, miny, maxx, maxy)):
                    # Warp geometries to requested CRS
                    shape = fiona.transform.transform_geom(
                        src.crs, self.crs.to_dict(), feature["geometry"]
                    )
                    if self.label_name:
                        shape = (shape, feature["properties"][self.label_name])
                    shapes.append(shape)

        # Rasterize geometries
        width = (query.maxx - query.minx) / self.res
        height = (query.maxy - query.miny) / self.res
        transform = rasterio.transform.from_bounds(
            query.minx, query.miny, query.maxx, query.maxy, width, height
        )
        if shapes:
            masks = rasterio.features.rasterize(
                shapes, out_shape=(round(height), round(width)), transform=transform
            )
        else:
            # If no features are found in this query, return an empty mask
            # with the default fill value and dtype used by rasterize
            masks = np.zeros((round(height), round(width)), dtype=np.uint8)

        sample = {"mask": torch.tensor(masks), "crs": self.crs, "bbox": query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample


class NonGeoDataset(Dataset[Dict[str, Any]], abc.ABC):
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
    type: NonGeoDataset
    size: {len(self)}"""


class VisionDataset(NonGeoDataset):
    """Abstract base class for datasets lacking geospatial information.

    .. deprecated:: 0.3
       Use :class:`NonGeoDataset` instead.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "VisionDataset":
        """Create a new instance of VisionDataset."""
        msg = "VisionDataset is deprecated, use NonGeoDataset instead."
        warnings.warn(msg, DeprecationWarning)
        return super().__new__(cls, *args, **kwargs)


class NonGeoClassificationDataset(NonGeoDataset, ImageFolder):  # type: ignore[misc]
    """Abstract base class for classification datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips which
    are separated into separate folders per class.
    """

    def __init__(
        self,
        root: str = "data",
        transforms: Optional[Callable[[Dict[str, Tensor]], Dict[str, Tensor]]] = None,
        loader: Optional[Callable[[str], Any]] = pil_loader,
        is_valid_file: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Initialize a new NonGeoClassificationDataset instance.

        Args:
            root: root directory where dataset can be found
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            loader: a callable function which takes as input a path to an image and
                returns a PIL Image or numpy array
            is_valid_file: A function that takes the path of an Image file and checks if
                the file is a valid file
        """
        # When transform & target_transform are None, ImageFolder.__getitem__(index)
        # returns a PIL.Image and int for image and label, respectively
        super().__init__(
            root=root,
            transform=None,
            target_transform=None,
            loader=loader,
            is_valid_file=is_valid_file,
        )

        # Must be set after calling super().__init__()
        self.transforms = transforms

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return
        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)
        sample = {"image": image, "label": label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int) -> Tuple[Tensor, Tensor]:
        """Load a single image and it's class label.

        Args:
            index: index to return
        Returns:
            the image
            the image class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        array: "np.typing.NDArray[np.int_]" = np.array(img)
        tensor = torch.from_numpy(array).float()
        # Convert from HxWxC to CxHxW
        tensor = tensor.permute((2, 0, 1))
        label = torch.tensor(label)
        return tensor, label


class VisionClassificationDataset(NonGeoClassificationDataset):
    """Abstract base class for classification datasets lacking geospatial information.

    .. deprecated:: 0.3
       Use :class:`NonGeoClassificationDataset` instead.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> "VisionClassificationDataset":
        """Create a new instance of VisionClassificationDataset."""
        msg = "VisionClassificationDataset is deprecated, "
        msg += "use NonGeoClassificationDataset instead."
        warnings.warn(msg, DeprecationWarning)
        return cast(VisionClassificationDataset, super().__new__(cls))


class IntersectionDataset(GeoDataset):
    """Dataset representing the intersection of two GeoDatasets.

    This allows users to do things like:

    * Combine image and target labels and sample from both simultaneously
      (e.g. Landsat and CDL)
    * Combine datasets for multiple image sources for multimodal learning or data fusion
      (e.g. Landsat and Sentinel)

    These combinations require that all queries are present in *both* datasets,
    and can be combined using an :class:`IntersectionDataset`:

    .. code-block:: python
       dataset = landsat & cdl

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        dataset1: GeoDataset,
        dataset2: GeoDataset,
        collate_fn: Callable[
            [Sequence[Dict[str, Any]]], Dict[str, Any]
        ] = concat_samples,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            dataset1: the first dataset
            dataset2: the second dataset
            collate_fn: function used to collate samples
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            ValueError: if either dataset is not a :class:`GeoDataset`

        .. versionadded:: 0.4
            The *transforms* parameter.
        """
        super().__init__(transforms)
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn

        for ds in self.datasets:
            if not isinstance(ds, GeoDataset):
                raise ValueError("IntersectionDataset only supports GeoDatasets")

        self._crs = dataset1.crs
        self.res = dataset1.res

        # Force dataset2 to have the same CRS/res as dataset1
        if dataset1.crs != dataset2.crs:
            print(
                f"Converting {dataset2.__class__.__name__} CRS from "
                f"{dataset2.crs} to {dataset1.crs}"
            )
            dataset2.crs = dataset1.crs
        if dataset1.res != dataset2.res:
            print(
                f"Converting {dataset2.__class__.__name__} resolution from "
                f"{dataset2.res} to {dataset1.res}"
            )
            dataset2.res = dataset1.res

        # Merge dataset indices into a single index
        self._merge_dataset_indices()

    def _merge_dataset_indices(self) -> None:
        """Create a new R-tree out of the individual indices from two datasets."""
        i = 0
        ds1, ds2 = self.datasets
        for hit1 in ds1.index.intersection(ds1.index.bounds, objects=True):
            for hit2 in ds2.index.intersection(hit1.bounds, objects=True):
                box1 = BoundingBox(*hit1.bounds)
                box2 = BoundingBox(*hit2.bounds)
                self.index.insert(i, tuple(box1 & box2))
                i += 1

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
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # All datasets are guaranteed to have a valid query
        samples = [ds[query] for ds in self.datasets]

        sample = self.collate_fn(samples)

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: IntersectionDataset
    bbox: {self.bounds}
    size: {len(self)}"""


class ForecastDataset(IntersectionDataset):
    """Dataset used for Forecasting tasks.

    This allows users to do things like:

    * Spatio-temporal predictions where input sequences come from one dataset
      and target sequences from another (e.g. Landsat and CDL)

    These combinations require that all queries are present in *both* datasets,
    and can be combined using an :class:`IntersectionDataset`:

    This dataset should be used with :class:'SequentialGeoSampler' sampler.

    .. versionadded:: 0.4
    """

    def __init__(
        self,
        input_dataset: GeoDataset,
        target_dataset: GeoDataset,
        collate_fn: Callable[
            [Sequence[Dict[str, Any]]], Dict[str, Any]
        ] = concat_samples,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            input_dataset: the input sequence dataset
            target_dataset: the target sequence dataset
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            ValueError: if either dataset is not a :class:`GeoDataset`
        """
        super().__init__(input_dataset, target_dataset, None, transforms)

        self.input_dataset = input_dataset
        self.target_dataset = target_dataset

    def __getitem__(self, query: Tuple[BoundingBox, BoundingBox]) -> Dict[str, Any]:
        """Retrieve image and metadata indexed by query.

        Args:
            query: two (minx, maxx, miny, maxy, mint, maxt) coordinates
                to index datasets, where first indexes the input dataset
                and second indexes the target dataset

        Returns:
            sample of data/targets and metadata at that index

        Raises:
            IndexError: if query is not within bounds of the index
        """
        if not query[0].intersects(self.bounds):
            raise IndexError(
                f"Input query: {query[0]} not found in input dataset "
                "index with bounds: {self.bounds}"
            )

        if not query[1].intersects(self.bounds):
            raise IndexError(
                f"Target query: {query[1]} not found in target dataset "
                "index with bounds: {self.bounds}"
            )
        # time-series where each query is for a different dataset
        # assuming 1-to-1 correspondence between query order and dataset order
        input_samples = self.input_dataset[query[0]]
        target_samples = self.target_dataset[query[1]]
        samples = {
            "input": input_samples["image"],
            "input_bbox": input_samples["bbox"],
            "input_crs": input_samples["crs"],
            "input_dates": input_samples["dates"],
            "target": target_samples["image"],
            "target_bbox": target_samples["bbox"],
            "target_crs": target_samples["crs"],
            "target_dates": target_samples["dates"],
        }

        return samples

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: ForecastDataset
    bbox: {self.bounds}
    size: {len(self)}
    input_dataset: {self.input_dataset}
    target_dataset: {self.target_dataset}"""

    def plot(self, input_sequence, target_sequence):
        """Plot sequential sample for input and target.

        Args:
            input_sequence: unbatched image sequence of shape
                num_time_steps, num_channels, height, width
            target_sequence: unbatched target sequence of shape
                num_time_steps, num_channels, height, width
        """

        def select_image(image, idx):
            return image[int(idx), 0, ...]

        fig, axs = plt.subplots(ncols=2)

        input_img = axs[0].imshow(select_image(input_sequence, 0))
        axs[0].set_title("Input Sequence")
        slider_ax = fig.add_axes([0.20, 0.1, 0.60, 0.03])
        input_slider = Slider(
            ax=slider_ax,
            label="Input Time Dimension",
            valmin=0,
            valmax=input_sequence.shape[0],
            valinit=0,
            valstep=1.0,
        )

        target_img = axs[1].imshow(select_image(target_sequence, 0))
        axs[1].set_title("Target Sequence")

        def update(val):
            input_img.array = select_image(input_sequence, 0)

            fig.canvas.draw_idle()

        input_slider.on_changed(update)
        plt.show()


class UnionDataset(GeoDataset):
    """Dataset representing the union of two GeoDatasets.

    This allows users to do things like:

    * Combine datasets for multiple image sources and treat them as equivalent
      (e.g. Landsat 7 and Landsat 8)
    * Combine datasets for disparate geospatial locations
      (e.g. Chesapeake NY and PA)

    These combinations require that all queries are present in *at least one* dataset,
    and can be combined using a :class:`UnionDataset`:

    .. code-block:: python

       dataset = landsat7 | landsat8

    .. versionadded:: 0.2
    """

    def __init__(
        self,
        dataset1: GeoDataset,
        dataset2: GeoDataset,
        collate_fn: Callable[
            [Sequence[Dict[str, Any]]], Dict[str, Any]
        ] = merge_samples,
        transforms: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            dataset1: the first dataset
            dataset2: the second dataset
            collate_fn: function used to collate samples
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            ValueError: if either dataset is not a :class:`GeoDataset`

        .. versionadded:: 0.4
            The *transforms* parameter.
        """
        super().__init__(transforms)
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn

        for ds in self.datasets:
            if not isinstance(ds, GeoDataset):
                raise ValueError("UnionDataset only supports GeoDatasets")

        self._crs = dataset1.crs
        self.res = dataset1.res

        # Force dataset2 to have the same CRS/res as dataset1
        if dataset1.crs != dataset2.crs:
            print(
                f"Converting {dataset2.__class__.__name__} CRS from "
                f"{dataset2.crs} to {dataset1.crs}"
            )
            dataset2.crs = dataset1.crs
        if dataset1.res != dataset2.res:
            print(
                f"Converting {dataset2.__class__.__name__} resolution from "
                f"{dataset2.res} to {dataset1.res}"
            )
            dataset2.res = dataset1.res

        # Merge dataset indices into a single index
        self._merge_dataset_indices()

    def _merge_dataset_indices(self) -> None:
        """Create a new R-tree out of the individual indices from two datasets."""
        i = 0
        for ds in self.datasets:
            hits = ds.index.intersection(ds.index.bounds, objects=True)
            for hit in hits:
                self.index.insert(i, hit.bounds)
                i += 1

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
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        # Not all datasets are guaranteed to have a valid query
        samples = []
        for ds in self.datasets:
            if list(ds.index.intersection(tuple(query))):
                samples.append(ds[query])

        sample = self.collate_fn(samples)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __str__(self) -> str:
        """Return the informal string representation of the object.

        Returns:
            informal string representation
        """
        return f"""\
{self.__class__.__name__} Dataset
    type: UnionDataset
    bbox: {self.bounds}
    size: {len(self)}"""
