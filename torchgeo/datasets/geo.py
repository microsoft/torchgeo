# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all :mod:`torchgeo` datasets."""

import abc
import functools
import glob
import os
import pathlib
import re
import sys
import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import Any, cast

import fiona
import fiona.transform
import numpy as np
import pyproj
import rasterio
import rasterio.merge
import shapely
import torch
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rtree.index import Index, Property
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader

from .errors import DatasetNotFoundError
from .utils import (
    BoundingBox,
    Path,
    array_to_tensor,
    concat_samples,
    disambiguate_timestamp,
    merge_samples,
    path_is_vsi,
    valid_data_footprint_from_datasource,
)


class GeoDataset(Dataset[dict[str, Any]], abc.ABC):
    """Abstract base class for datasets containing geospatial information.

    Geospatial information includes things like:

    * coordinates (latitude, longitude)
    * :term:`coordinate reference system (CRS)`
    * resolution

    :class:`GeoDataset` is a special class of datasets. Unlike :class:`NonGeoDataset`,
    the presence of geospatial information allows two or more datasets to be combined
    based on latitude/longitude. This allows users to do things like:

    * Combine image and target labels and sample from both simultaneously
      (e.g., Landsat and CDL)
    * Combine datasets for multiple image sources for multimodal learning or data fusion
      (e.g., Landsat and Sentinel)
    * Combine image and other raster data (e.g., elevation, temperature, pressure)
      and sample from both simultaneously (e.g., Landsat and Aster Global DEM)

    These combinations require that all queries are present in *both* datasets,
    and can be combined using an :class:`IntersectionDataset`:

    .. code-block:: python

       dataset = landsat & cdl

    Users may also want to:

    * Combine datasets for multiple image sources and treat them as equivalent
      (e.g., Landsat 7 and Landsat 8)
    * Combine datasets for disparate geospatial locations
      (e.g., Chesapeake NY and PA)

    These combinations require that all queries are present in *at least one* dataset,
    and can be combined using a :class:`UnionDataset`:

    .. code-block:: python

       dataset = landsat7 | landsat8
    """

    paths: Path | Iterable[Path]
    _crs = CRS.from_epsg(4326)
    _res = 0.0

    #: Glob expression used to search for files.
    #:
    #: This expression should be specific enough that it will not pick up files from
    #: other datasets. It should not include a file extension, as the dataset may be in
    #: a different file format than what it was originally downloaded as.
    filename_glob = '*'

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
        self, transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None
    ) -> None:
        """Initialize a new GeoDataset instance.

        Args:
            transforms: a function/transform that takes an input sample
                and returns a transformed version
        """
        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

    @abc.abstractmethod
    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """

    def __and__(self, other: 'GeoDataset') -> 'IntersectionDataset':
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

    def __or__(self, other: 'GeoDataset') -> 'UnionDataset':
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

    def __getstate__(self) -> tuple[dict[str, Any], list[tuple[Any, Any, Any | None]]]:
        """Define how instances are pickled.

        Returns:
            the state necessary to unpickle the instance
        """
        objects = self.index.intersection(self.index.bounds, objects=True)
        tuples = [(item.id, item.bounds, item.object) for item in objects]
        return self.__dict__, tuples

    def __setstate__(
        self,
        state: tuple[
            dict[Any, Any],
            list[tuple[int, tuple[float, float, float, float, float, float], Path]],
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
        """:term:`coordinate reference system (CRS)` of the dataset.

        Returns:
            The :term:`coordinate reference system (CRS)`.
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of a GeoDataset.

        If ``new_crs == self.crs``, does nothing, otherwise updates the R-tree index.

        Args:
            new_crs: New :term:`coordinate reference system (CRS)`.
        """
        if new_crs == self.crs:
            return

        print(f'Converting {self.__class__.__name__} CRS from {self.crs} to {new_crs}')
        new_index = Index(interleaved=False, properties=Property(dimension=3))

        project = pyproj.Transformer.from_crs(
            pyproj.CRS(str(self.crs)), pyproj.CRS(str(new_crs)), always_xy=True
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

    @property
    def res(self) -> float:
        """Resolution of the dataset in units of CRS.

        Returns:
            The resolution of the dataset.
        """
        return self._res

    @res.setter
    def res(self, new_res: float) -> None:
        """Change the resolution of a GeoDataset.

        Args:
            new_res: New resolution.
        """
        if new_res == self.res:
            return

        print(f'Converting {self.__class__.__name__} res from {self.res} to {new_res}')
        self._res = new_res

    @property
    def files(self) -> list[Path]:
        """A list of all files in the dataset.

        Returns:
            All files in the dataset.

        .. versionadded:: 0.5
        """
        # Make iterable
        if isinstance(self.paths, str | pathlib.Path):
            paths: Iterable[Path] = [self.paths]
        else:
            paths = self.paths

        # Using set to remove any duplicates if directories are overlapping
        files: set[Path] = set()
        for path in paths:
            if os.path.isdir(path):
                pathname = os.path.join(path, '**', self.filename_glob)
                files |= set(glob.iglob(pathname, recursive=True))
            elif os.path.isfile(path) or path_is_vsi(path):
                files.add(path)
            elif not hasattr(self, 'download'):
                warnings.warn(
                    f"Could not find any relevant files for provided path '{path}'. "
                    f'Path was ignored.',
                    UserWarning,
                )

        # Sort the output to enforce deterministic behavior.
        return sorted(files)


class RasterDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as raster files."""

    #: Regular expression used to extract date from filename.
    #:
    #: The expression should use named groups. The expression may contain any number of
    #: groups. The following groups are specifically searched for by the base class:
    #:
    #: * ``date``: used to calculate ``mint`` and ``maxt`` for ``index`` insertion
    #: * ``start``: used to calculate ``mint`` for ``index`` insertion
    #: * ``stop``: used to calculate ``maxt`` for ``index`` insertion
    #:
    #: When :attr:`~RasterDataset.separate_files` is True, the following additional
    #: groups are searched for to find other files:
    #:
    #: * ``band``: replaced with requested band name
    filename_regex = '.*'

    #: Date format string used to parse date from filename.
    #:
    #: Not used if :attr:`filename_regex` does not contain a ``date`` group or
    #: ``start`` and ``stop`` groups.
    date_format = '%Y%m%d'

    #: Minimum timestamp if not in filename
    mint: float = 0

    #: Maximum timestmap if not in filename
    maxt: float = sys.maxsize

    #: True if the dataset only contains model inputs (such as images). False if the
    #: dataset only contains ground truth model outputs (such as segmentation masks).
    #:
    #: The sample returned by the dataset/data loader will use the "image" key if
    #: *is_image* is True, otherwise it will use the "mask" key.
    #:
    #: For datasets with both model inputs and outputs, a custom
    #: :func:`~RasterDataset.__getitem__` method must be implemented.
    is_image = True

    #: True if data is stored in a separate file for each band, else False.
    separate_files = False

    #: Names of all available bands in the dataset
    all_bands: list[str] = []

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: list[str] = []

    #: Color map for the dataset, used for plotting
    cmap: dict[int, tuple[int, int, int, int]] = {}

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the dataset (overrides the dtype of the data file via a cast).

        Defaults to float32 if :attr:`~RasterDataset.is_image` is True, else long.
        Can be overridden for tasks like pixel-wise regression where the mask should be
        float32 instead of long.

        Returns:
            the dtype of the dataset

        .. versionadded:: 0.5
        """
        if self.is_image:
            return torch.float32
        else:
            return torch.long

    @property
    def resampling(self) -> Resampling:
        """Resampling algorithm used when reading input files.

        Defaults to bilinear for float dtypes and nearest for int dtypes.

        Returns:
            The resampling method to use.

        .. versionadded:: 0.6
        """
        # Based on torch.is_floating_point
        if self.dtype in [torch.float64, torch.float32, torch.float16, torch.bfloat16]:
            return Resampling.bilinear
        else:
            return Resampling.nearest

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        cache: bool = True,
    ) -> None:
        """Initialize a new RasterDataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
                (defaults to the resolution of the first file found)
            bands: bands to return (defaults to all bands)
            transforms: a function/transform that takes an input sample
                and returns a transformed version
            cache: if True, cache file handle to speed up repeated sampling

        Raises:
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        super().__init__(transforms)

        self.paths = paths
        self.bands = bands or self.all_bands
        self.cache = cache

        # Populate the dataset index
        i = 0
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in self.files:
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

                        valid_footprint = valid_data_footprint_from_datasource(src, crs)

                        with WarpedVRT(src, crs=crs) as vrt:
                            minx, miny, maxx, maxy = vrt.bounds
                            if res is None:
                                res = vrt.res[0]
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    mint = self.mint
                    maxt = self.maxt
                    if 'date' in match.groupdict():
                        date = match.group('date')
                        mint, maxt = disambiguate_timestamp(date, self.date_format)
                    elif 'start' in match.groupdict() and 'stop' in match.groupdict():
                        start = match.group('start')
                        stop = match.group('stop')
                        mint, _ = disambiguate_timestamp(start, self.date_format)
                        _, maxt = disambiguate_timestamp(stop, self.date_format)

                    coords = (minx, maxx, miny, maxy, mint, maxt)

                    self.index.insert(
                        i,
                        coords,
                        {"filepath": filepath, "valid_footprint": valid_footprint},
                    )
                    i += 1

        if i == 0:
            raise DatasetNotFoundError(self)

        if not self.separate_files:
            self.band_indexes = None
            if self.bands:
                if self.all_bands:
                    self.band_indexes = [
                        self.all_bands.index(i) + 1 for i in self.bands
                    ]
                else:
                    msg = (
                        f'{self.__class__.__name__} is missing an `all_bands` '
                        'attribute, so `bands` cannot be specified.'
                    )
                    raise AssertionError(msg)

        self._crs = cast(CRS, crs)
        self._res = cast(float, res)

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(
            list[Path], [cast(dict[str, Any], hit.object)["filepath"] for hit in hits]
        )

        if not filepaths:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        if self.separate_files:
            data_list: list[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in filepaths:
                    filename = os.path.basename(filepath)
                    directory = os.path.dirname(filepath)
                    match = re.match(filename_regex, filename)
                    if match:
                        if 'band' in match.groupdict():
                            start = match.start('band')
                            end = match.end('band')
                            filename = filename[:start] + band + filename[end:]
                    filepath = os.path.join(directory, filename)
                    band_filepaths.append(filepath)
                data_list.append(self._merge_files(band_filepaths, query))
            data = torch.cat(data_list)
        else:
            data = self._merge_files(filepaths, query, self.band_indexes)

        sample = {'crs': self.crs, 'bounds': query}

        data = data.to(self.dtype)
        if self.is_image:
            sample['image'] = data
        else:
            sample['mask'] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(
        self,
        filepaths: Sequence[Path],
        query: BoundingBox,
        band_indexes: Sequence[int] | None = None,
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index
            band_indexes: indexes of bands to be used

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        bounds = (query.minx, query.miny, query.maxx, query.maxy)
        dest, _ = rasterio.merge.merge(
            vrt_fhs, bounds, self.res, indexes=band_indexes, resampling=self.resampling
        )
        # Use array_to_tensor since merge may return uint16/uint32 arrays.
        tensor = array_to_tensor(dest)
        return tensor

    @functools.lru_cache(maxsize=128)
    def _cached_load_warp_file(self, filepath: Path) -> DatasetReader:
        """Cached version of :meth:`_load_warp_file`.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        return self._load_warp_file(filepath)

    def _load_warp_file(self, filepath: Path) -> DatasetReader:
        """Load and warp a file to the correct CRS and resolution.

        Args:
            filepath: file to load and warp

        Returns:
            file handle of warped VRT
        """
        # Todo: need to update meta kwarg `nodata` if it is not set.
        #  create memory-file?
        #  https://rasterio.groups.io/g/main/topic/change_the_nodata_value_in_a/28801885?p=
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            valid_nodatavals = [val for val in src.nodatavals if val is not None]
            if not valid_nodatavals:  # Case for Sentinel2 L1C
                nodata = 0.0  # Is it safe to assume? For Sentinel2 L1C it is 0.0
            else:
                # Get the first valid nodata value.
                # Usualy the same value for all bands
                nodata = valid_nodatavals[0]
            vrt = WarpedVRT(src, nodata=nodata, crs=self.crs)
            src.close()
            return vrt
        else:
            return src


class VectorDataset(GeoDataset):
    """Abstract base class for :class:`GeoDataset` stored as vector files."""

    #: Regular expression used to extract date from filename.
    #:
    #: The expression should use named groups. The expression may contain any number of
    #: groups. The following groups are specifically searched for by the base class:
    #:
    #: * ``date``: used to calculate ``mint`` and ``maxt`` for ``index`` insertion
    filename_regex = '.*'

    #: Date format string used to parse date from filename.
    #:
    #: Not used if :attr:`filename_regex` does not contain a ``date`` group.
    date_format = '%Y%m%d'

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the dataset (overrides the dtype of the data file via a cast).

        Defaults to long.

        Returns:
            the dtype of the dataset

        .. versionadded:: 0.6
        """
        return torch.long

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float = 0.0001,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        label_name: str | None = None,
    ) -> None:
        """Initialize a new VectorDataset instance.

        Args:
            paths: one or more root directories to search or files to load
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the first file found)
            res: resolution of the dataset in units of CRS
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version
            label_name: name of the dataset property that has the label to be
                rasterized into the mask

        Raises:
            DatasetNotFoundError: If dataset is not found.

        .. versionadded:: 0.4
            The *label_name* parameter.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        super().__init__(transforms)

        self.paths = paths
        self.label_name = label_name

        # Populate the dataset index
        i = 0
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in self.files:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
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
                    mint: float = 0
                    maxt: float = sys.maxsize
                    if 'date' in match.groupdict():
                        date = match.group('date')
                        mint, maxt = disambiguate_timestamp(date, self.date_format)
                    coords = (minx, maxx, miny, maxy, mint, maxt)
                    self.index.insert(i, coords, filepath)
                    i += 1

        if i == 0:
            raise DatasetNotFoundError(self)

        self._crs = crs
        self._res = res

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
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
                f'query: {query} not found in index with bounds: {self.bounds}'
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
                        src.crs, self.crs.to_dict(), feature['geometry']
                    )
                    label = self.get_label(feature)
                    shapes.append((shape, label))

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

        # Use array_to_tensor since rasterize may return uint16/uint32 arrays.
        masks = array_to_tensor(masks)

        masks = masks.to(self.dtype)
        sample = {'mask': masks, 'crs': self.crs, 'bounds': query}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def get_label(self, feature: 'fiona.model.Feature') -> int:
        """Get label value to use for rendering a feature.

        Args:
            feature: the :class:`fiona.model.Feature` from which to extract the label.

        Returns:
            the integer label, or 0 if the feature should not be rendered.

        .. versionadded:: 0.6
        """
        if self.label_name:
            return int(feature['properties'][self.label_name])
        return 1


class NonGeoDataset(Dataset[dict[str, Any]], abc.ABC):
    """Abstract base class for datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips.
    """

    @abc.abstractmethod
    def __getitem__(self, index: int) -> dict[str, Any]:
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


class NonGeoClassificationDataset(NonGeoDataset, ImageFolder):  # type: ignore[misc]
    """Abstract base class for classification datasets lacking geospatial information.

    This base class is designed for datasets with pre-defined image chips which
    are separated into separate folders per class.
    """

    def __init__(
        self,
        root: Path = 'data',
        transforms: Callable[[dict[str, Tensor]], dict[str, Tensor]] | None = None,
        loader: Callable[[Path], Any] | None = pil_loader,
        is_valid_file: Callable[[Path], bool] | None = None,
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

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        """Return an index within the dataset.

        Args:
            index: index to return

        Returns:
            data and label at that index
        """
        image, label = self._load_image(index)
        sample = {'image': image, 'label': label}

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def __len__(self) -> int:
        """Return the number of data points in the dataset.

        Returns:
            length of the dataset
        """
        return len(self.imgs)

    def _load_image(self, index: int) -> tuple[Tensor, Tensor]:
        """Load a single image and its class label.

        Args:
            index: index to return

        Returns:
            the image and class label
        """
        img, label = ImageFolder.__getitem__(self, index)
        array: np.typing.NDArray[np.int_] = np.array(img)
        tensor = torch.from_numpy(array).float()
        # Convert from HxWxC to CxHxW
        tensor = tensor.permute((2, 0, 1))
        label = torch.tensor(label)
        return tensor, label


class IntersectionDataset(GeoDataset):
    """Dataset representing the intersection of two GeoDatasets.

    This allows users to do things like:

    * Combine image and target labels and sample from both simultaneously
      (e.g., Landsat and CDL)
    * Combine datasets for multiple image sources for multimodal learning or data fusion
      (e.g., Landsat and Sentinel)
    * Combine image and other raster data (e.g., elevation, temperature, pressure)
      and sample from both simultaneously (e.g., Landsat and Aster Global DEM)

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
            [Sequence[dict[str, Any]]], dict[str, Any]
        ] = concat_samples,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a new IntersectionDataset instance.

        When computing the intersection between two datasets that both contain model
        inputs (such as images) or model outputs (such as masks), the default behavior
        is to stack the data along the channel dimension. The *collate_fn* parameter
        can be used to change this behavior.

        Args:
            dataset1: the first dataset
            dataset2: the second dataset
            collate_fn: function used to collate samples
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            RuntimeError: if datasets have no spatiotemporal intersection
            ValueError: if either dataset is not a :class:`GeoDataset`

        .. versionadded:: 0.4
            The *transforms* parameter.
        """
        super().__init__(transforms)
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn

        for ds in self.datasets:
            if not isinstance(ds, GeoDataset):
                raise ValueError('IntersectionDataset only supports GeoDatasets')

        self.crs = dataset1.crs
        self.res = dataset1.res

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
                box_intersection = box1 & box2
                # Skip 0 area overlap (unless 0 area dataset)
                if box_intersection.area > 0 or box1.area == 0 or box2.area == 0:
                    # assuming hit1 is the rasterfile.
                    # It might have nodata-regions covered by other files,
                    # which RasterData will read from.
                    # we merge the footprint from all files covering this bounds
                    all_footprints_overlapping = [
                        other.object["valid_footprint"]
                        for other in ds1.index.intersection(ds1.index.bounds, objects=True)
                    ]
                    all_footprints_cropped_to_bounds = shapely.intersection(
                        shapely.ops.unary_union(all_footprints_overlapping),
                        shapely.geometry.box(
                            box_intersection.minx,
                            box_intersection.miny,
                            box_intersection.maxx,
                            box_intersection.maxy,
                        ),
                    )
                    self.index.insert(
                        i,
                        tuple(box_intersection),
                        cast(
                            dict[str, shapely.geometry.Polygon],
                            all_footprints_cropped_to_bounds,
                        ),
                    )
                i += 1

        if i == 0:
            raise RuntimeError('Datasets have no spatiotemporal intersection')

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
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
                f'query: {query} not found in index with bounds: {self.bounds}'
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

    @property
    def crs(self) -> CRS:
        """:term:`coordinate reference system (CRS)` of both datasets.

        Returns:
            The :term:`coordinate reference system (CRS)`.
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of both datasets.

        Args:
            new_crs: New :term:`coordinate reference system (CRS)`.
        """
        self._crs = new_crs
        self.datasets[0].crs = new_crs
        self.datasets[1].crs = new_crs

    @property
    def res(self) -> float:
        """Resolution of both datasets in units of CRS.

        Returns:
            Resolution of both datasets.
        """
        return self._res

    @res.setter
    def res(self, new_res: float) -> None:
        """Change the resolution of both datasets.

        Args:
            new_res: New resolution.
        """
        self._res = new_res
        self.datasets[0].res = new_res
        self.datasets[1].res = new_res


class UnionDataset(GeoDataset):
    """Dataset representing the union of two GeoDatasets.

    This allows users to do things like:

    * Combine datasets for multiple image sources and treat them as equivalent
      (e.g., Landsat 7 and Landsat 8)
    * Combine datasets for disparate geospatial locations
      (e.g., Chesapeake NY and PA)

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
            [Sequence[dict[str, Any]]], dict[str, Any]
        ] = merge_samples,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a new UnionDataset instance.

        When computing the union between two datasets that both contain model inputs
        (such as images) or model outputs (such as masks), the default behavior is to
        merge the data to create a single image/mask. The *collate_fn* parameter can be
        used to change this behavior.

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
                raise ValueError('UnionDataset only supports GeoDatasets')

        self.crs = dataset1.crs
        self.res = dataset1.res

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

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
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
                f'query: {query} not found in index with bounds: {self.bounds}'
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

    @property
    def crs(self) -> CRS:
        """:term:`coordinate reference system (CRS)` of both datasets.

        Returns:
            The :term:`coordinate reference system (CRS)`.
        """
        return self._crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of both datasets.

        Args:
            new_crs: New :term:`coordinate reference system (CRS)`.
        """
        self._crs = new_crs
        self.datasets[0].crs = new_crs
        self.datasets[1].crs = new_crs

    @property
    def res(self) -> float:
        """Resolution of both datasets in units of CRS.

        Returns:
            The resolution of both datasets.
        """
        return self._res

    @res.setter
    def res(self, new_res: float) -> None:
        """Change the resolution of both datasets.

        Args:
            new_res: New resolution.
        """
        self._res = new_res
        self.datasets[0].res = new_res
        self.datasets[1].res = new_res
