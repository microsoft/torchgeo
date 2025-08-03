# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Base classes for all :mod:`torchgeo` datasets."""

import abc
import fnmatch
import functools
import glob
import os
import re
import warnings
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from typing import Any, ClassVar, Literal

import xarray as xr
from rasterio.crs import CRS
from rioxarray.merge import merge_arrays
from rtree.index import Index, Property

import fiona
import fiona.transform
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.merge
import shapely
import torch
from geopandas import GeoDataFrame
from pyproj import CRS
from rasterio.enums import Resampling
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader as pil_loader

from .errors import DatasetNotFoundError
from .utils import (
    GeoSlice,
    Path,
    array_to_tensor,
    concat_samples,
    convert_poly_coords,
    disambiguate_timestamp,
    merge_samples,
    path_is_vsi,
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

    index: GeoDataFrame
    paths: Path | Iterable[Path]
    _res = (0.0, 0.0)

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

    def _disambiguate_slice(self, query: GeoSlice) -> tuple[slice, slice, slice]:
        """Disambiguate a partial spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            A fully resolved spatiotemporal slice.
        """
        out = list(self.bounds)

        if isinstance(query, slice):
            query = (query,)

        # For each slice (x, y, t)...
        for i in range(len(query)):
            # For each component (start, stop, step)...
            if query[i].start is not None:
                out[i] = slice(query[i].start, out[i].stop, out[i].step)
            if query[i].stop is not None:
                out[i] = slice(out[i].start, query[i].stop, out[i].step)
            if query[i].step is not None:
                out[i] = slice(out[i].start, out[i].stop, query[i].step)

        geoslice = tuple(out)
        assert len(geoslice) == 3
        return geoslice

    @abc.abstractmethod
    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *query* is not found in the index.
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

    @property
    def bounds(self) -> tuple[slice, slice, slice]:
        """Bounds of the index.

        Returns:
            Bounding x, y, and t slices.
        """
        xmin, ymin, xmax, ymax = self.index.total_bounds
        xres, yres = self.res
        tmin = self.index.index.left.min()
        tmax = self.index.index.right.max()
        tres = 1
        return slice(xmin, xmax, xres), slice(ymin, ymax, yres), slice(tmin, tmax, tres)

    @property
    def crs(self) -> CRS:
        """:term:`coordinate reference system (CRS)` of the dataset.

        Returns:
            The :term:`coordinate reference system (CRS)`.
        """
        _crs: CRS = self.index.crs
        return _crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of a GeoDataset.

        If ``new_crs == self.crs``, does nothing, otherwise updates the index.

        Args:
            new_crs: New :term:`coordinate reference system (CRS)`.
        """
        if new_crs == self.crs:
            return

        print(f'Converting {self.__class__.__name__} CRS from {self.crs} to {new_crs}')
        self.index.to_crs(new_crs, inplace=True)

    @property
    def res(self) -> tuple[float, float]:
        """Resolution of the dataset in units of CRS.

        Returns:
            The resolution of the dataset.
        """
        return self._res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        """Change the resolution of a GeoDataset.

        Args:
            new_res: New resolution in (xres, yres) format. If a single float is provided, it is used for both
                the x and y resolution.
        """
        if isinstance(new_res, int | float):
            new_res = (new_res, new_res)

        if new_res == self.res:
            return

        print(f'Converting {self.__class__.__name__} res from {self.res} to {new_res}')
        self._res = new_res

    @property
    def files(self) -> list[str]:
        """A list of all files in the dataset.

        Returns:
            All files in the dataset.

        .. versionadded:: 0.5
        """
        # Make iterable
        if isinstance(self.paths, str | os.PathLike):
            paths: Iterable[Path] = [self.paths]
        else:
            paths = self.paths

        # Using set to remove any duplicates if directories are overlapping
        files: set[str] = set()
        for path in paths:
            if os.path.isdir(path):
                pathname = os.path.join(path, '**', self.filename_glob)
                files |= set(glob.iglob(pathname, recursive=True))
            elif (os.path.isfile(path) or path_is_vsi(path)) and fnmatch.fnmatch(
                str(path), f'*{self.filename_glob}'
            ):
                files.add(str(path))
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
    mint: datetime = pd.Timestamp.min

    #: Maximum timestamp if not in filename
    maxt: datetime = pd.Timestamp.max

    #: True if the dataset only contains model inputs (such as images). False if the
    #: dataset only contains ground truth model outputs (such as segmentation masks).
    #:
    #: The sample returned by the dataset/data loader will use the "image" key if
    #: *is_image* is True, otherwise it will use the "mask" key.
    #:
    #: For datasets with both model inputs and outputs, the recommended approach is
    #: to use 2 `RasterDataset` instances and combine them using an `IntersectionDataset`.
    is_image = True

    #: True if data is stored in a separate file for each band, else False.
    separate_files = False

    #: Names of all available bands in the dataset
    all_bands: tuple[str, ...] = ()

    #: Names of RGB bands in the dataset, used for plotting
    rgb_bands: tuple[str, ...] = ()

    #: Color map for the dataset, used for plotting
    cmap: ClassVar[dict[int, tuple[int, int, int, int]]] = {}

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
        res: float | tuple[float, float] | None = None,
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
            AssertionError: If *bands* are invalid.
            DatasetNotFoundError: If dataset is not found.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.
        """
        self.paths = paths
        self.bands = bands or self.all_bands
        self.transforms = transforms
        self.cache = cache

        if self.all_bands:
            assert set(self.bands) <= set(self.all_bands)

        # Gather information about the dataset
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        filepaths = []
        datetimes = []
        geometries = []
        for filepath in self.files:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with rasterio.open(filepath) as src:
                        # See if file has a color map
                        if len(self.cmap) == 0:
                            try:
                                self.cmap = src.colormap(1)  # type: ignore[misc]
                            except ValueError:
                                pass

                        if crs is None:
                            crs = src.crs

                        with WarpedVRT(src, crs=crs) as vrt:
                            geometries.append(shapely.box(*vrt.bounds))
                            if res is None:
                                res = vrt.res
                except rasterio.errors.RasterioIOError:
                    # Skip files that rasterio is unable to read
                    continue
                else:
                    filepaths.append(filepath)

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

                    datetimes.append((mint, maxt))

        if len(filepaths) == 0:
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

        if res is not None:
            if isinstance(res, int | float):
                res = (res, res)

            self._res = res

        # Create the dataset index
        data = {'filepath': filepaths}
        index = pd.IntervalIndex.from_tuples(datetimes, closed='both', name='datetime')
        self.index = GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *query* is not found in the index.
        """
        x, y, t = self._disambiguate_slice(query)
        interval = pd.Interval(t.start, t.stop)
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.iloc[:: t.step]
        index = index.cx[x.start : x.stop, y.start : y.stop]

        if index.empty:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        if self.separate_files:
            data_list: list[Tensor] = []
            filename_regex = re.compile(self.filename_regex, re.VERBOSE)
            for band in self.bands:
                band_filepaths = []
                for filepath in index.filepath:
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
            data = self._merge_files(index.filepath, query, self.band_indexes)

        sample: dict[str, Any] = {'crs': self.crs, 'bounds': query}

        data = data.to(self.dtype)
        if self.is_image:
            sample['image'] = data
        else:
            sample['mask'] = data.squeeze(0)

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample

    def _merge_files(
        self,
        filepaths: Sequence[str],
        query: GeoSlice,
        band_indexes: Sequence[int] | None = None,
    ) -> Tensor:
        """Load and merge one or more files.

        Args:
            filepaths: one or more files to load and merge
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.
            band_indexes: indexes of bands to be used

        Returns:
            image/mask at that index
        """
        if self.cache:
            vrt_fhs = [self._cached_load_warp_file(fp) for fp in filepaths]
        else:
            vrt_fhs = [self._load_warp_file(fp) for fp in filepaths]

        x, y, t = self._disambiguate_slice(query)
        bounds = (x.start, y.start, x.stop, y.stop)
        res = (x.step, y.step)
        dest, _ = rasterio.merge.merge(
            vrt_fhs, bounds, res, indexes=band_indexes, resampling=self.resampling
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
        src = rasterio.open(filepath)

        # Only warp if necessary
        if src.crs != self.crs:
            vrt = WarpedVRT(src, crs=self.crs)
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
        res: float | tuple[float, float] = (0.0001, 0.0001),
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        label_name: str | None = None,
        task: Literal[
            'object_detection', 'semantic_segmentation', 'instance_segmentation'
        ] = 'semantic_segmentation',
        layer: str | int | None = None,
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
            task: computer vision task the dataset is used for. Supported output types
               `object_detection`, `semantic_segmentation`, `instance_segmentation`
            layer: if the input is a multilayer vector dataset, such as a geopackage,
                specify which layer to use. Can be int to specify the index of the layer,
                str to select the layer with that name or None which opens the first layer

        Raises:
            DatasetNotFoundError: If dataset is not found.
            ValueError: If task is not one of allowed tasks

        .. versionadded:: 0.4
            The *label_name* parameter.

        .. versionchanged:: 0.5
           *root* was renamed to *paths*.

        .. versionadded:: 0.8
           The *task* and *layer* parameters
        """
        self.paths = paths
        self.transforms = transforms
        self.label_name = label_name
        # List of allowed tasks
        allowed_tasks = [
            'semantic_segmentation',
            'object_detection',
            'instance_segmentation',
        ]
        if task not in allowed_tasks:
            raise ValueError(f'Invalid task: {task!r}. Must be one of {allowed_tasks}')
        self.task = task
        self.layer = layer
        # Gather information about the dataset
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        filepaths = []
        datetimes = []
        geometries = []
        for filepath in self.files:
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                try:
                    with fiona.open(filepath, layer=layer) as src:
                        if crs is None:
                            crs = CRS.from_wkt(src.crs_wkt)

                        minx, miny, maxx, maxy = src.bounds
                        (minx, maxx), (miny, maxy) = fiona.transform.transform(
                            src.crs, crs.to_wkt(), [minx, maxx], [miny, maxy]
                        )
                        geometry = shapely.box(minx, miny, maxx, maxy)
                        geometries.append(geometry)
                except fiona.errors.FionaValueError:
                    # Skip files that fiona is unable to read
                    continue
                else:
                    filepaths.append(filepath)

                    mint = pd.Timestamp.min
                    maxt = pd.Timestamp.max
                    if 'date' in match.groupdict():
                        date = match.group('date')
                        mint, maxt = disambiguate_timestamp(date, self.date_format)

                    datetimes.append((mint, maxt))

        if len(filepaths) == 0:
            raise DatasetNotFoundError(self)

        if isinstance(res, int | float):
            res = (res, res)

        self._res = res

        # Create the dataset index
        data = {'filepath': filepaths}
        index = pd.IntervalIndex.from_tuples(datetimes, closed='both', name='datetime')
        self.index = GeoDataFrame(data, index=index, geometry=geometries, crs=crs)

    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *query* is not found in the index.
        """
        x, y, t = self._disambiguate_slice(query)
        interval = pd.Interval(t.start, t.stop)
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.iloc[:: t.step]
        index = index.cx[x.start : x.stop, y.start : y.stop]

        if index.empty:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

        shapes = []
        for filepath in index.filepath:
            with fiona.open(filepath, layer=self.layer) as src:
                # We need to know the bounding box of the query in the source CRS
                (minx, maxx), (miny, maxy) = fiona.transform.transform(
                    self.crs.to_wkt(), src.crs, [x.start, x.stop], [y.start, y.stop]
                )

                # Filter geometries to those that intersect with the bounding box
                for feature in src.filter(bbox=(minx, miny, maxx, maxy)):
                    # Warp geometries to requested CRS
                    shape = fiona.transform.transform_geom(
                        src.crs, self.crs.to_wkt(), feature['geometry']
                    )
                    label = self.get_label(feature)
                    shapes.append((shape, label))

        # Rasterize geometries
        width = (x.stop - x.start) / x.step
        height = (y.stop - y.start) / y.step
        transform = rasterio.transform.from_bounds(
            x.start, y.start, x.stop, y.stop, width, height
        )
        if shapes:
            match self.task:
                case 'semantic_segmentation':
                    masks = rasterio.features.rasterize(
                        shapes,
                        out_shape=(round(height), round(width)),
                        transform=transform,
                    )

                case 'object_detection':
                    # Get boxes for object detection or instance segmentation
                    px_shapes = [
                        convert_poly_coords(
                            shapely.geometry.shape(s[0]), transform, inverse=True
                        )
                        for s in shapes
                    ]
                    px_shapes = [
                        (shapely.clip_by_rect(p, 0, 0, width, height))
                        for p in px_shapes
                    ]

                    # Get labels
                    labels = np.array([s[1] for s in shapes]).astype(np.int32)

                    # xmin, ymin, xmax, ymax format
                    boxes_xyxy = np.array(
                        [
                            [p.bounds[0], p.bounds[1], p.bounds[2], p.bounds[3]]
                            for p in px_shapes
                        ]
                    ).astype(np.float32)

                case 'instance_segmentation':
                    # Get boxes for object detection or instance segmentation
                    px_shapes = [
                        convert_poly_coords(
                            shapely.geometry.shape(s[0]), transform, inverse=True
                        )
                        for s in shapes
                    ]
                    px_shapes = [
                        (shapely.clip_by_rect(p, 0, 0, width, height))
                        for p in px_shapes
                    ]

                    # Get labels
                    labels = np.array([s[1] for s in shapes]).astype(np.int32)

                    # xmin, ymin, xmax, ymax format
                    boxes_xyxy = np.array(
                        [
                            [p.bounds[0], p.bounds[1], p.bounds[2], p.bounds[3]]
                            for p in px_shapes
                        ]
                    ).astype(np.float32)

                    masks = rasterio.features.rasterize(
                        [(s[0], i + 1) for i, s in enumerate(shapes)],
                        out_shape=(round(height), round(width)),
                        transform=transform,
                    )

                    obj_ids = np.unique(masks)

                    # first id is the background, so remove it
                    obj_ids = obj_ids[1:]

                    # convert (H, W) mask a set of binary masks
                    masks = (masks == obj_ids[:, None, None]).astype(np.uint8)
        else:
            # If no features are found in this key, return an empty mask
            # with the default fill value and dtype used by rasterize
            masks = np.zeros((round(height), round(width)), dtype=np.uint8)
            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            labels = np.empty((0,), dtype=np.int32)

        # Use array_to_tensor since rasterize may return uint16/uint32 arrays.
        sample: dict[str, Any] = {'crs': self.crs, 'bounds': query}

        match self.task:
            case 'semantic_segmentation':
                sample['mask'] = array_to_tensor(masks).to(self.dtype)

            case 'object_detection':
                sample['bbox_xyxy'] = torch.from_numpy(boxes_xyxy)
                sample['label'] = torch.from_numpy(labels)

            case 'instance_segmentation':
                sample['mask'] = array_to_tensor(masks)
                sample['bbox_xyxy'] = torch.from_numpy(boxes_xyxy)
                sample['label'] = torch.from_numpy(labels)

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
        spatial_only: bool = False,
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
            spatial_only: if True, ignore temporal dimension when computing intersection
            collate_fn: function used to collate samples
            transforms: a function/transform that takes input sample and its target as
                entry and returns a transformed version

        Raises:
            RuntimeError: if datasets have no spatiotemporal intersection
            ValueError: if either dataset is not a :class:`GeoDataset`

        .. versionadded:: 0.8
           The *spatial_only* parameter.

        .. versionadded:: 0.4
           The *transforms* parameter.
        """
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn
        self.transforms = transforms

        for ds in self.datasets:
            if not isinstance(ds, GeoDataset):
                raise ValueError('IntersectionDataset only supports GeoDatasets')

        dataset2.crs = dataset1.crs
        dataset2.res = dataset1.res

        # Spatial intersection
        index1 = dataset1.index.reset_index()
        index2 = dataset2.index.reset_index()
        self.index = gpd.overlay(
            index1, index2, how='intersection', keep_geom_type=True
        )

        if self.index.empty:
            raise RuntimeError('Datasets have no spatial intersection')

        # Temporal intersection
        if not spatial_only:
            datetime_1 = pd.IntervalIndex(self.index.pop('datetime_1'))
            datetime_2 = pd.IntervalIndex(self.index.pop('datetime_2'))
            mint = np.maximum(datetime_1.left, datetime_2.left)
            maxt = np.minimum(datetime_1.right, datetime_2.right)
            valid = maxt >= mint
            mint = mint[valid]
            maxt = maxt[valid]
            self.index = self.index[valid]
            self.index.index = pd.IntervalIndex.from_arrays(
                mint, maxt, closed='both', name='datetime'
            )

            if self.index.empty:
                msg = 'Datasets have no temporal intersection. Use `spatial_only=True`'
                msg += ' if you want to ignore temporal intersection'
                raise RuntimeError(msg)

    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *query* is not found in the index.
        """
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
        return self.datasets[0].crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of both datasets.

        Args:
            new_crs: New :term:`coordinate reference system (CRS)`.
        """
        self.index.to_crs(new_crs, inplace=True)
        self.datasets[0].crs = new_crs
        self.datasets[1].crs = new_crs

    @property
    def res(self) -> tuple[float, float]:
        """Resolution of both datasets in units of CRS.

        Returns:
            Resolution of both datasets.
        """
        return self.datasets[0].res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        """Change the resolution of both datasets.

        Args:
            new_res: New resolution.
        """
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
        self.datasets = [dataset1, dataset2]
        self.collate_fn = collate_fn
        self.transforms = transforms

        for ds in self.datasets:
            if not isinstance(ds, GeoDataset):
                raise ValueError('UnionDataset only supports GeoDatasets')

        dataset2.crs = dataset1.crs
        dataset2.res = dataset1.res

        self.index = pd.concat([dataset1.index, dataset2.index])

    def __getitem__(self, query: GeoSlice) -> dict[str, Any]:
        """Retrieve input, target, and/or metadata indexed by spatiotemporal slice.

        Args:
            query: [xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres] coordinates to index.

        Returns:
            Sample of input, target, and/or metadata at that index.

        Raises:
            IndexError: If *query* is not found in the index.
        """
        # Not all datasets are guaranteed to have a valid query
        samples = []
        for ds in self.datasets:
            try:
                samples.append(ds[query])
            except IndexError:
                pass

        if not samples:
            raise IndexError(
                f'query: {query} not found in index with bounds: {self.bounds}'
            )

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
        return self.datasets[0].crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Change the :term:`coordinate reference system (CRS)` of both datasets.

        Args:
            new_crs: New :term:`coordinate reference system (CRS)`.
        """
        self.index.to_crs(new_crs, inplace=True)
        self.datasets[0].crs = new_crs
        self.datasets[1].crs = new_crs

    @property
    def res(self) -> tuple[float, float]:
        """Resolution of both datasets in units of CRS.

        Returns:
            The resolution of both datasets.
        """
        return self.datasets[0].res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        """Change the resolution of both datasets.

        Args:
            new_res: New resolution.
        """
        self.datasets[0].res = new_res
        self.datasets[1].res = new_res



class RioXarrayDataset(GeoDataset):
    """Wrapper for geographical datasets stored as Xarray Datasets.
    
    In-memory geographical xarray.DataArray and xarray.Dataset.

    Relies on rioxarray.

    .. versionadded:: 0.7.0
    """

    filename_glob = "*"
    filename_regex = ".*"

    is_image = True

    spatial_x_name = "x"
    spatial_y_name = "y"

    transform = None

    @property
    def dtype(self) -> torch.dtype:
        """The dtype of the dataset (overrides the dtype of the data file via a cast).

        Returns:
            the dtype of the dataset
        """
        if self.is_image:
            return torch.float32
        else:
            return torch.long

    def harmonize_format(self, ds):
        """Convert the dataset to the standard format.

        Args:
            ds: dataset or array to harmonize

        Returns:
            the harmonized dataset or array
        """
        # rioxarray expects spatial dimensions to be named x and y
        ds.rio.set_spatial_dims(self.spatial_x_name, self.spatial_y_name, inplace=True)

        # if x coords go from 0 to 360, convert to -180 to 180
        if ds[self.spatial_x_name].min() > 180:
            ds = ds.assign_coords(
                {self.spatial_x_name: ds[self.spatial_x_name] % 360 - 180}
            )

        # if y coords go from 0 to 180, convert to -90 to 90
        if ds[self.spatial_x_name].min() > 90:
            ds = ds.assign_coords(
                {self.spatial_y_name: ds[self.spatial_y_name] % 180 - 90}
            )
        # expect asceding coordinate values
        ds = ds.sortby(self.spatial_x_name, ascending=True)
        ds = ds.sortby(self.spatial_y_name, ascending=True)
        return ds

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        data_variables: list[str] | None = None,
        # crs: Optional[CRS] = None,
        transforms: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        """Initialize a new Dataset instance.

        Args:
            paths: one or more root directories to search or files to load
            data_variables: data variables that should be gathered from the collection
                of xarray datasets
            transforms: a function/transform that takes an input sample
                and returns a transformed version

        Raises:
            FileNotFoundError: if files are not found in ``paths``
        """
        super().__init__(transforms)

        self.paths = paths

        if data_variables:
            self.data_variables = data_variables
        else:
            data_variables_to_collect: list[str] = []

        self.transforms = transforms

        # Create an R-tree to index the dataset
        self.index = Index(interleaved=False, properties=Property(dimension=3))

        # Populate the dataset index
        i = 0
        pathname = os.path.join(root, self.filename_glob)
        filename_regex = re.compile(self.filename_regex, re.VERBOSE)
        for filepath in glob.iglob(pathname, recursive=True):
            match = re.match(filename_regex, os.path.basename(filepath))
            if match is not None:
                with xr.open_dataset(filepath, decode_times=True) as ds:
                    ds = self.harmonize_format(ds)

                    try:
                        (minx, miny, maxx, maxy) = ds.rio.bounds()
                    except AttributeError:
                        # or take the shape of the data variable?
                        continue

                if hasattr(ds, "time"):
                    try:
                        indices = ds.indexes["time"].to_datetimeindex()
                    except AttributeError:
                        indices = ds.indexes["time"]

                    mint = indices.min().to_pydatetime().timestamp()
                    maxt = indices.max().to_pydatetime().timestamp()
                else:
                    mint = 0
                    maxt = sys.maxsize
                coords = (minx, maxx, miny, maxy, mint, maxt)
                self.index.insert(i, coords, filepath)
                i += 1

                # collect all possible data variables if self.data_variables is None
                if not data_variables:
                    data_variables_to_collect.extend(list(ds.data_vars))

        if i == 0:
            import pdb

            pdb.set_trace()
            msg = f"No {self.__class__.__name__} data was found in `paths='{self.paths}'`"
            raise FileNotFoundError(msg)

        if not data_variables:
            self.data_variables = list(set(data_variables_to_collect))

        # if not crs:
        #     self._crs = "EPSG:4326"
        # else:
        #     self._crs = cast(CRS, crs)
        self.res = 1.0

    def _infer_spatial_coordinate_names(self, ds) -> tuple[str]:
        """Infer the names of the spatial coordinates.

        Args:
            ds: Dataset or DataArray of which to infer the spatial coordinates

        Returns:
            x and y coordinate names
        """
        x_name = None
        y_name = None
        for coord_name, coord in ds.coords.items():
            if hasattr(coord, "units"):
                if any(
                    [
                        x in coord.units.lower()
                        for x in ["degrees_north", "degree_north"]
                    ]
                ):
                    y_name = coord_name
                elif any(
                    [x in coord.units.lower() for x in ["degrees_east", "degree_east"]]
                ):
                    x_name = coord_name

        if not x_name or not y_name:
            raise ValueError("Spatial Coordinate Units not found in Dataset.")

        return x_name, y_name

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
        items = [hit.object for hit in hits]

        if not items:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data_arrays: list["np.typing.NDArray"] = []
        for item in items:
            with xr.open_dataset(item, decode_cf=True) as ds:
                ds = self.harmonize_format(ds)
                # select time dimension
                if hasattr(ds, "time"):
                    try:
                        ds["time"] = ds.indexes["time"].to_datetimeindex()
                    except AttributeError:
                        ds["time"] = ds.indexes["time"]
                    ds = ds.sel(
                        time=slice(
                            datetime.fromtimestamp(query.mint),
                            datetime.fromtimestamp(query.maxt),
                        )
                    )

                for variable in self.data_variables:
                    if hasattr(ds, variable):
                        da = ds[variable]
                        # if not da.rio.crs:
                        #     da.rio.write_crs(self._crs, inplace=True)
                        # elif da.rio.crs != self._crs:
                        #     da = da.rio.reproject(self._crs)
                        # clip box ignores time dimension
                        clipped = da.rio.clip_box(
                            minx=query.minx,
                            miny=query.miny,
                            maxx=query.maxx,
                            maxy=query.maxy,
                        )
                        # rioxarray expects this order
                        clipped = clipped.transpose(
                            "time", self.spatial_y_name, self.spatial_x_name, ...
                        )

                        # set proper transform # TODO not working
                        # clipped.rio.write_transform(self.transform)
                        data_arrays.append(clipped.squeeze())

        import pdb

        pdb.set_trace()
        merged_data = torch.from_numpy(
            merge_arrays(
                data_arrays, bounds=(query.minx, query.miny, query.maxx, query.maxy)
            ).data
        )
        sample = {"bbox": query}

        merged_data = merged_data.to(self.dtype)
        if self.is_image:
            sample["image"] = merged_data
        else:
            sample["mask"] = merged_data

        if self.transforms is not None:
            sample = self.transforms(sample)

        return sample