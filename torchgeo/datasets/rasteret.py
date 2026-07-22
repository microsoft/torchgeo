# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Rasteret dataset."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import torch
from geopandas import GeoDataFrame
from pyproj import CRS

from torchgeo.datasets.geo import RasterDataset, Sample
from torchgeo.datasets.utils import GeoSlice, lazy_import


class RasteretDataset(RasterDataset):
    """A dataset backed by a `Rasteret <https://github.com/terrafloww/rasteret>`_ collection.

    `Rasteret <https://github.com/terrafloww/rasteret>`_ is a library for fast,
    cloud-native reads of Cloud-Optimized GeoTIFFs (COGs) catalogued with STAC.
    Rather than filesystem paths, this dataset is created from a Rasteret
    ``Collection`` -- an index of scenes and their COG assets, usually built with
    ``rasteret.build(...)`` and reopened with ``rasteret.load(...)``.

    The sampling CRS and resolution come from the collection's stored metadata,
    so no raster is opened until a patch is sampled; each query is then read
    from the COGs and mosaicked onto the requested grid. Pass ``crs`` or ``res``
    to override the defaults (the CRS is fixed once the dataset is created).

    For multiprocessing (``DataLoader(num_workers=...)``), use a collection
    opened with ``rasteret.load(...)``; an in-memory collection cannot be sent
    to worker processes.

    Example:
        >>> import rasteret
        >>> from torchgeo.datasets import RasteretDataset
        >>> collection = rasteret.load('sentinel2_collection')  # doctest: +SKIP
        >>> dataset = RasteretDataset(
        ...     collection, bands=['B04', 'B03', 'B02']
        ... )  # doctest: +SKIP

    .. warning::
       This dataset is experimental and its API may change.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        collection: Any,
        bands: Sequence[str],
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        time_series: bool = False,
        is_image: bool = True,
        max_concurrent: int = 50,
        backend: Any = None,
    ) -> None:
        """Initialize a new RasteretDataset instance.

        Args:
            collection: A Rasteret ``Collection``, typically from
                ``rasteret.build(...)``, ``rasteret.build_from_stac(...)``,
                ``rasteret.build_from_table(...)``, or ``rasteret.load(...)``.
            bands: Band codes to load (e.g. ``('B04', 'B03', 'B02')``).
            crs: Sampling CRS. Must be EPSG-resolvable. Defaults to the
                collection's native CRS.
            res: Sampling resolution in CRS units. Defaults to the band's
                native resolution read from collection metadata.
            transforms: A function/transform that takes an input sample and
                returns a transformed version.
            cache: Accepted for ``RasterDataset`` compatibility; not used.
            time_series: If ``True``, stack overlapping items along a leading
                time dimension instead of mosaicking to a single image.
            is_image: If ``True``, return values in ``sample['image']``.
            max_concurrent: Maximum concurrent HTTP byte-range requests per read.
            backend: Optional Rasteret ``StorageBackend`` for authenticated or
                requester-pays buckets.

        Raises:
            ValueError: If no bands are given, *crs* is not EPSG-resolvable, the
                collection has mixed CRS with no *crs* override, or it yields no
                footprints.
            TypeError: If *collection* is not a Rasteret ``Collection`` (missing
                ``footprints``, ``native_res``, or ``read_window``).
        """
        lazy_import('rasteret')

        if not bands:
            raise ValueError('At least one band is required')
        for method in ('footprints', 'native_res', 'read_window'):
            if not callable(getattr(collection, method, None)):
                raise TypeError(
                    f'collection must be a rasteret.Collection exposing {method}(...)'
                )

        target_crs: int | None = None
        if crs is not None:
            epsg = crs.to_epsg()
            if epsg is None:
                raise ValueError('RasteretDataset requires an EPSG CRS')
            target_crs = int(epsg)

        # No local file paths: reads are served from the collection, not disk.
        self._collection = collection
        self.paths = ''
        self.bands = tuple(bands)
        self.all_bands = tuple(bands)
        self.transforms = transforms
        self.cache = cache
        self.time_series = time_series
        self.is_image = is_image
        self.separate_files = False
        self.band_indexes = None
        self._max_concurrent = max_concurrent
        self._backend = backend

        # Index each scene by its exact COG footprint in the sampling CRS.
        # Reprojecting the collection's stored WGS84 bounds instead would inflate
        # them and cause scenes to match queries they do not actually cover.
        footprints = collection.footprints(crs=target_crs, band=self.bands[0])
        if footprints.empty:
            raise ValueError('Rasteret collection produced no footprints')
        if footprints.crs is None:
            raise ValueError('Collection has mixed CRS; pass crs= to unify.')
        epsg = CRS.from_user_input(footprints.crs).to_epsg()
        if epsg is None:
            raise ValueError(
                'Collection CRS is not EPSG-resolvable; pass an EPSG crs=.'
            )
        self._target_crs = int(epsg)

        if res is not None:
            self._res = (
                (float(res), float(res))
                if isinstance(res, (int, float))
                else (float(res[0]), float(res[1]))
            )
        else:
            self._res = collection.native_res(self.bands[0])

        # Sort for a stable sample order. Rasteret also reads this order as
        # mosaic priority: where scenes overlap, earlier ones win.
        order = footprints.sort_values(['datetime', 'id']).index
        footprints = footprints.loc[order]
        dt = pd.to_datetime(footprints['datetime'], utc=True)
        self.index = GeoDataFrame(
            {'id': footprints['id'].to_list()},
            index=pd.IntervalIndex.from_arrays(dt, dt, closed='both', name='datetime'),
            geometry=footprints.geometry.to_numpy(),
            crs=self._target_crs,
        )

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve a sample indexed by spatiotemporal slice.

        Args:
            index: ``[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]`` query.

        Returns:
            Sample at the requested index.

        Raises:
            IndexError: If no collection records intersect the query.
        """
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop, closed='both')
        matches = self.index.iloc[self.index.index.overlaps(interval)][:: t.step]
        matches = matches.cx[x.start : x.stop, y.start : y.stop]
        if matches.empty:
            raise IndexError(
                f'index: {index} not found in dataset with bounds: {self.bounds}'
            )

        array = self._collection.read_window(
            record_ids=matches['id'].to_list(),
            bounds=(x.start, y.start, x.stop, y.stop),
            res=(x.step, y.step),
            bands=list(self.bands),
            target_crs=self._target_crs,
            max_concurrent=self._max_concurrent,
            backend=self._backend,
            group_by='id' if self.time_series else None,
        )

        sample: Sample = {
            'image': torch.from_numpy(np.ascontiguousarray(array)).to(self.dtype),
            'bounds': self._slice_to_tensor(index),
            'transform': torch.tensor(
                rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
            ),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the dataset.

        Returns:
            The dataset CRS.
        """
        return CRS.from_epsg(self._target_crs)

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Reject post-init CRS changes.

        Args:
            new_crs: New CRS.

        Raises:
            AttributeError: Always raised. Rasteret binds the read-time CRS at
                construction; use ``RasteretDataset(..., crs=...)`` instead.
        """
        raise AttributeError(
            'RasteretDataset CRS is fixed after construction; '
            'create a new dataset with crs=...'
        )

    @property
    def res(self) -> tuple[float, float]:
        """Resolution of the dataset in units of CRS.

        Returns:
            The dataset resolution.
        """
        return self._res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        """Change dataset resolution.

        Args:
            new_res: New resolution as a scalar or ``(xres, yres)`` tuple.
        """
        self._res = (
            (float(new_res), float(new_res))
            if isinstance(new_res, (int, float))
            else (float(new_res[0]), float(new_res[1]))
        )

    @property
    def dtype(self) -> torch.dtype:
        """The dtype used for outputs.

        Returns:
            ``torch.float32`` for imagery and ``torch.long`` for masks.
        """
        if self.is_image:
            return torch.float32
        else:
            return torch.long

    @property
    def files(self) -> list[str]:
        """COG asset hrefs backing the dataset, in collection order.

        Rasteret reads from a collection rather than local paths, so this lists
        the remote COG URLs for the requested bands instead of filesystem files.

        Returns:
            Deduplicated asset hrefs, or an empty list if the collection exposes
            no ``assets`` column.
        """
        try:
            table = self._collection.to_table(columns=['assets'])
        except Exception:
            return []
        hrefs: list[str] = []
        for assets in table['assets'].to_pylist():
            if not isinstance(assets, dict):
                continue
            for band in self.bands:
                asset = assets.get(band)
                if isinstance(asset, dict) and asset.get('href'):
                    hrefs.append(str(asset['href']))
        return list(dict.fromkeys(hrefs))

    def close(self) -> None:
        """Release Rasteret background resources if the collection supports it."""
        close = getattr(self._collection, 'close', None)
        if callable(close):
            close()
