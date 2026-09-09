# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""WeatherBench 2 dataset."""

import glob
import os
from collections.abc import Callable, Iterable, Sequence
from contextlib import nullcontext
from typing import Any, cast

import numpy as np
import pandas as pd
import shapely
import torch
from geopandas import GeoDataFrame
from pyproj import CRS
from torch import Tensor

from .errors import DatasetNotFoundError
from .geo import XarrayDataset
from .utils import GeoSlice, Path, Sample, lazy_import


class WeatherBench2(XarrayDataset):
    """WeatherBench 2 dataset.

    `WeatherBench <https://sites.research.google/gr/weatherbench/>`__ is an open
    framework for evaluating ML and physics-based weather forecasting models in a
    like-for-like fashion.

    This data loader supports several publicly available, cloud-optimized ground-truth
    and baseline `datasets
    <https://weatherbench2.readthedocs.io/en/latest/data-guide.html>`__,
    including a comprehensive copy of the
    `ERA5 <https://rmets.onlinelibrary.wiley.com/doi/full/10.1002/qj.3803>`__
    dataset used for training most ML models.

    Stores are read with :func:`xarray.open_zarr`, which supports both local Zarr
    directories and remote object stores (e.g. ``gs://weatherbench2/...``) via
    `fsspec`_ and `gcsfs`_. ``paths`` may therefore be:

    * a single local ``*.zarr`` directory,
    * a directory containing one or more ``*.zarr`` stores,
    * a ``gs://`` (or other ``://``) URI pointing at a public Zarr store.

    For **public** ``gs://`` buckets (such as WeatherBench2 on GCS), pass
    ``storage_options={'token': 'anon'}`` when you do not have Google
    Application Default Credentials

    See the
    `documentation <https://weatherbench2.readthedocs.io/en/latest/data-guide.html>`__
    for more information on data availability.

    Each sample contains:

    * ``image``: tensor stacked from the requested ``data_vars`` in order, only
      populated when all selected variables share the same shape. When mixing
      surface, pressure-level, and static variables, this key is omitted;
      consume ``variables`` instead.
    * ``variables``: mapping from variable name to its tensor, with the
      original (possibly heterogeneous) per-variable shape preserved.
    * ``lat`` / ``lon``: 1-D coordinate tensors for the spatial slice.
    * ``time``: tuple of selected timestamps.
    * ``atmos_levels``: tuple of pressure levels if any were selected.
    * ``bounds`` and ``transform``: standard TorchGeo metadata used by
      :class:`~torchgeo.datasets.GeoDataset`.

    .. note::

       This dataset requires the following additional libraries to be installed:

       * `xarray <https://pypi.org/project/xarray/>`_
       * `zarr <https://pypi.org/project/zarr/>`_
       * `fsspec <https://pypi.org/project/fsspec/>`_
       * `gcsfs <https://pypi.org/project/gcsfs/>`_ (only for ``gs://`` paths)

    If you use this dataset in your research, please cite the following paper:

    * https://arxiv.org/abs/2308.15560

    .. _fsspec: https://filesystem-spec.readthedocs.io/
    .. _gcsfs: https://gcsfs.readthedocs.io/

    .. versionadded:: 0.10
    """

    filename_glob = '*.zarr'

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        data_vars: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        storage_options: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a new WeatherBench2 instance.

        Args:
            paths: a local Zarr store, a directory containing Zarr stores, or a
                remote URI (``gs://...``) pointing to a Zarr store.
            crs: :term:`coordinate reference system (CRS)` to assume for the
                store (defaults to ``EPSG:4326``, which matches WeatherBench2).
            res: resolution of the store in degrees, as a single value or
                ``(xres, yres)`` tuple. Inferred from ``longitude``/``latitude``
                coordinates when omitted.
            data_vars: list of variable names to load. Defaults to all
                ``data_vars`` of the first store found.
            transforms: a function/transform that takes an input sample and
                returns a transformed version.
            storage_options: optional keyword arguments forwarded to
                :func:`xarray.open_zarr` (for example ``{'token': 'anon'}`` for
                public GCS data without Application Default Credentials).

        Raises:
            DatasetNotFoundError: If no Zarr stores are found at *paths*.
            DependencyNotFoundError: If xarray/zarr is not installed.
        """
        xr = lazy_import('xarray')
        lazy_import('zarr')
        self.paths = paths
        self.transforms = transforms
        self.storage_options = storage_options

        filepaths: list[str] = []
        datetimes: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        geometries: list[shapely.Polygon] = []
        for filepath in self.files:
            try:
                with self._open_xarray(
                    filepath, xr, storage_options=self.storage_options
                ) as src:
                    _crs, _res, _bounds = self._spatial_metadata(src, crs, res)
                    crs, res = _crs, _res
                    data_vars = data_vars or list(src.data_vars.keys())
                    tmin = pd.Timestamp(src.time.values.min())
                    tmax = pd.Timestamp(src.time.values.max())
                    filepaths.append(filepath)
                    datetimes.append((tmin, tmax))
                    geometries.append(shapely.box(*_bounds))
            except (OSError, ValueError):
                # Skip stores that xarray is unable to read or that are missing
                # the spatial/temporal coordinates we rely on.
                continue

        if not filepaths:
            raise DatasetNotFoundError(self)

        # ``res`` is normalized to a tuple inside :meth:`_spatial_metadata`
        # and a missing value would have raised :class:`DatasetNotFoundError`,
        # so it must be a populated tuple here.
        assert isinstance(res, tuple)
        self._res = res
        self.data_vars = list(data_vars) if data_vars is not None else []
        self.index = GeoDataFrame(
            {'filepath': filepaths},
            index=pd.IntervalIndex.from_tuples(
                datetimes, closed='both', name='datetime'
            ),
            geometry=geometries,
            crs=crs,
        )

    @property
    def files(self) -> list[str]:
        """A list of all Zarr stores in the dataset.

        Unlike :attr:`~torchgeo.datasets.GeoDataset.files`, this implementation
        treats ``*.zarr`` directories as opaque stores instead of descending into
        them, and accepts remote ``gs://`` URIs.

        Returns:
            All Zarr stores in the dataset.
        """
        if isinstance(self.paths, str | os.PathLike):
            paths: Iterable[Path] = [cast(Path, self.paths)]
        else:
            paths = self.paths

        files: set[str] = set()
        for path in paths:
            spath = str(path)
            if '://' in spath:
                # Remote URI (e.g. gs://...).
                files.add(spath)
            elif os.path.isdir(spath) and spath.endswith('.zarr'):
                files.add(spath)
            elif os.path.isdir(spath):
                pattern = os.path.join(spath, '**', self.filename_glob)
                files |= set(glob.iglob(pattern, recursive=True))
        return sorted(files)

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve a sample for a spatiotemporal slice.

        Args:
            index: ``[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]`` slice.

        Returns:
            Sample dict described in the class docstring.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        xr = lazy_import('xarray')

        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f'index: {index} not found in dataset with bounds: {self.bounds}'
            )

        ds, lon_name, lat_name, levels = self._open_and_slice(df.filepath, index, xr)
        variables = self._dataset_to_tensors(ds)
        lat = torch.as_tensor(np.asarray(ds[lat_name].values, dtype=float))
        lon = torch.as_tensor(np.asarray(ds[lon_name].values, dtype=float))
        time = tuple(pd.Timestamp(v) for v in np.atleast_1d(ds.time.values))

        sample: Sample = {
            'bounds': self._slice_to_tensor(index),
            'variables': variables,
            'lat': lat,
            'lon': lon,
            'time': time,
            'atmos_levels': levels,
            'transform': torch.tensor(
                [x.step, 0.0, x.start, 0.0, -y.step, y.stop, 0.0, 0.0, 1.0]
            ).reshape(3, 3),
        }
        # Only populate ``image`` when all selected variables share a shape;
        # mixed surface/pressure-level/static variables cannot be stacked, so
        # callers should consume ``variables`` directly in that case.
        shapes = {tuple(v.shape) for v in variables.values()}
        if len(shapes) == 1:
            sample['image'] = torch.stack(list(variables.values()))
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @staticmethod
    def _open_xarray(
        path: str, xr: Any | None = None, storage_options: dict[str, Any] | None = None
    ) -> Any:
        """Open a Zarr store as an :class:`xarray.Dataset`.

        Lazy-imports :mod:`fsspec` and :mod:`gcsfs` for ``gs://`` paths so the
        dependency is only required when remote streaming is actually used.

        Args:
            path: local Zarr directory or ``gs://`` URI.
            xr: optional cached ``xarray`` module (avoids re-importing).
            storage_options: optional arguments for :func:`xarray.open_zarr`
                (passed to ``fsspec`` / ``gcsfs`` for remote stores).

        Returns:
            A context manager yielding the opened :class:`xarray.Dataset`.
        """
        if xr is None:
            xr = lazy_import('xarray')
        is_remote = '://' in str(path)
        if is_remote:
            lazy_import('fsspec')
            if str(path).startswith('gs://'):
                lazy_import('gcsfs')
        open_kw: dict[str, Any] = {}
        if storage_options is not None and is_remote:
            open_kw['storage_options'] = storage_options
        ds = xr.open_zarr(path, **open_kw)
        return nullcontext(ds)

    @staticmethod
    def _coord_names(src: Any) -> tuple[str, str]:
        """Return the names of the longitude and latitude coordinates."""
        lon_name = next(
            (n for n in ('longitude', 'lon') if n in src.coords), 'longitude'
        )
        lat_name = next((n for n in ('latitude', 'lat') if n in src.coords), 'latitude')
        return lon_name, lat_name

    @classmethod
    def _spatial_metadata(
        cls, src: Any, crs: CRS | None, res: float | tuple[float, float] | None
    ) -> tuple[CRS, tuple[float, float], tuple[float, float, float, float]]:
        """Infer CRS, resolution, and bounds from ``latitude``/``longitude`` coords.

        WeatherBench2 stores are tagged in plain ``EPSG:4326`` and do not carry
        ``rioxarray`` metadata. We therefore default to ``EPSG:4326`` and infer
        resolution and bounds from the coordinate arrays.
        """
        lon_name, lat_name = cls._coord_names(src)
        if lon_name not in src.coords or lat_name not in src.coords:
            raise ValueError('Store is missing longitude/latitude coordinates.')

        lon = np.asarray(src[lon_name].values, dtype=float)
        lat = np.asarray(src[lat_name].values, dtype=float)
        if lon.size < 2 or lat.size < 2:
            raise ValueError('Store has fewer than 2 longitude/latitude values.')

        if res is None:
            xres = float(abs(lon[1] - lon[0]))
            yres = float(abs(lat[1] - lat[0]))
            res = (xres, yres)
        elif isinstance(res, int | float):
            res = (float(res), float(res))

        xmin, xmax = float(lon.min()), float(lon.max())
        ymin, ymax = float(lat.min()), float(lat.max())
        return crs or CRS.from_epsg(4326), res, (xmin, ymin, xmax, ymax)

    def _open_and_slice(
        self, filepaths: Sequence[str], index: GeoSlice, xr: Any
    ) -> tuple[Any, str, str, tuple[float, ...]]:
        """Open the matching Zarr stores and apply a coordinate slice.

        Returns:
            ``(dataset, lon_name, lat_name, levels)`` where ``levels`` is the
            tuple of pressure levels in the slice (empty if not present).
        """
        x, y, t = self._disambiguate_slice(index)

        # WeatherBench2 ships as a single store in practice. Concatenate along
        # the existing time dimension when more than one store is selected.
        contexts = [
            self._open_xarray(fp, xr, storage_options=self.storage_options)
            for fp in filepaths
        ]
        with contexts[0] as first:
            srcs = [first]
            for ctx in contexts[1:]:
                with ctx as more:
                    srcs.append(more)
            # Pin ``data_vars='all'`` so we stay compatible with both the
            # current xarray default and the upcoming change to ``None``.
            ds = (
                srcs[0]
                if len(srcs) == 1
                else xr.concat(srcs, dim='time', data_vars='all')
            )

            lon_name, lat_name = self._coord_names(ds)
            lat_vals = np.asarray(ds[lat_name].values, dtype=float)
            lon_vals = np.asarray(ds[lon_name].values, dtype=float)

            # Latitude is often stored in descending order in WeatherBench2.
            lat_slice = (
                slice(y.stop, y.start)
                if lat_vals[0] > lat_vals[-1]
                else slice(y.start, y.stop)
            )
            lon_slice = (
                slice(x.stop, x.start)
                if lon_vals[0] > lon_vals[-1]
                else slice(x.start, x.stop)
            )

            ds = ds.sel({lon_name: lon_slice, lat_name: lat_slice})
            ds = ds.sel(time=slice(t.start, t.stop))
            # Materialize so we don't rely on a context manager at sample time.
            ds = ds.load()

        levels: tuple[float, ...] = ()
        if 'level' in ds.coords:
            levels = tuple(float(v) for v in np.atleast_1d(ds['level'].values))
        return ds, lon_name, lat_name, levels

    def _dataset_to_tensors(self, ds: Any) -> dict[str, Tensor]:
        """Materialize selected ``data_vars`` as float32 tensors."""
        out: dict[str, Tensor] = {}
        for var in self.data_vars:
            if var not in ds.data_vars:
                continue
            arr = np.asarray(ds[var].values, dtype=np.float32)
            out[var] = torch.from_numpy(arr)
        return out
