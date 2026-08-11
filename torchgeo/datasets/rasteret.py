# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Rasteret dataset."""

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import pandas as pd
import rasterio
import torch
from geopandas import GeoDataFrame
from pyproj import CRS

from torchgeo.datasets.geo import RasterDataset, Sample
from torchgeo.datasets.utils import GeoSlice, array_to_tensor, lazy_import

if TYPE_CHECKING:
    from rasteret.cloud import StorageBackend
    from rasteret.core.collection import Collection


class RasteretDataset(RasterDataset):
    """A dataset backed by a `Rasteret <https://github.com/terrafloww/rasteret>`_ collection.

    `Rasteret <https://github.com/terrafloww/rasteret>`_ is a library for fast,
    cloud-native reads of Cloud-Optimized GeoTIFFs (COGs) catalogued with STAC.
    Rather than filesystem paths, this dataset is created from a Rasteret
    ``Collection`` -- an index of scenes and their COG assets, usually built with
    ``rasteret.build(...)`` and reopened with ``rasteret.load(...)``.

    The sampling CRS and resolution come from the collection's stored metadata,
    so no raster is opened until a patch is sampled; each query is then read from
    the COGs and mosaicked onto the requested grid. As with ``RasterDataset``,
    pass ``crs`` or ``res`` to sample onto a different grid.

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
        collection: 'Collection',
        bands: Sequence[str] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        time_series: bool = False,
        is_image: bool = True,
        max_concurrent: int = 50,
        backend: 'StorageBackend | None' = None,
    ) -> None:
        """Initialize a new RasteretDataset instance.

        Args:
            collection: A Rasteret ``Collection``, e.g. from ``rasteret.build(...)``
                or ``rasteret.load(...)``.
            bands: Band codes to load. Defaults to every band in the collection.
            crs: Sampling CRS. Defaults to the collection's native CRS.
            res: Sampling resolution in CRS units. Defaults to the band's native
                resolution read from collection metadata.
            transforms: A function/transform that takes an input sample and returns
                a transformed version.
            cache: Accepted for ``RasterDataset`` compatibility; not used.
            time_series: If ``True``, stack overlapping scenes along a leading time
                dimension instead of mosaicking to a single image.
            is_image: If ``True``, return values in ``sample['image']``.
            max_concurrent: Maximum concurrent HTTP byte-range requests per read.
            backend: Optional Rasteret ``StorageBackend`` for authenticated or
                requester-pays buckets.
        """
        lazy_import('rasteret')

        self.collection = collection
        self.bands = tuple(bands) if bands else tuple(collection.bands)
        self.all_bands = tuple(collection.bands)
        self.transforms = transforms
        self.cache = cache
        self.time_series = time_series
        self.is_image = is_image
        self.max_concurrent = max_concurrent
        self.backend = backend
        # Reads are served from the collection, not disk, so there are no file paths.
        self.paths = []

        # Index each scene by its exact COG footprint in the sampling CRS. Reprojecting
        # the collection's stored WGS84 bounds instead would inflate them and match
        # queries a scene does not actually cover.
        epsg = crs.to_epsg() if crs is not None else None
        footprints = collection.footprints(crs=epsg, band=self.bands[0])

        res = res if res is not None else collection.native_res(self.bands[0])
        self._res = (res, res) if isinstance(res, int | float) else tuple(res)

        # Sort for a stable sample order. Rasteret also reads this order as mosaic
        # priority: where scenes overlap, earlier ones win.
        footprints = footprints.sort_values(['datetime', 'id'])
        dt = pd.to_datetime(footprints['datetime'], utc=True)
        self.index = GeoDataFrame(
            {'id': footprints['id'].to_list()},
            index=pd.IntervalIndex.from_arrays(dt, dt, closed='both', name='datetime'),
            geometry=footprints.geometry.to_numpy(),
            crs=footprints.crs,
        )

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve a sample indexed by spatiotemporal slice.

        Args:
            index: ``[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]`` query.

        Returns:
            Sample at the requested index.

        Raises:
            IndexError: If *index* is not found in the dataset.
        """
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop, closed='both')
        matches = self.index.iloc[self.index.index.overlaps(interval)][:: t.step]
        matches = matches.cx[x.start : x.stop, y.start : y.stop]
        if matches.empty:
            raise IndexError(
                f'index: {index} not found in dataset with bounds: {self.bounds}'
            )

        array = self.collection.read_window(
            record_ids=matches['id'].to_list(),
            bounds=(x.start, y.start, x.stop, y.stop),
            res=(x.step, y.step),
            bands=list(self.bands),
            target_crs=self.crs.to_epsg(),
            max_concurrent=self.max_concurrent,
            backend=self.backend,
            group_by='id' if self.time_series else None,
        )

        sample: Sample = {
            # array_to_tensor (as RasterDataset does) casts the uint16/uint32 arrays
            # that torch.from_numpy rejects, without precision loss.
            'image': array_to_tensor(array).to(self.dtype),
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
        """:term:`coordinate reference system (CRS)` of the dataset.

        Returns:
            The dataset CRS.
        """
        return cast(CRS, self.index.crs)

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Reject post-construction CRS changes.

        Unlike ``RasterDataset``, Rasteret binds the read-time CRS when the
        collection index is built, so reprojecting the index afterwards would
        silently disagree with what is read. Pass ``crs=`` at construction.

        Args:
            new_crs: New CRS (ignored).

        Raises:
            AttributeError: Always, to prevent a silent index/read CRS mismatch.
        """
        raise AttributeError('RasteretDataset CRS is fixed at construction; pass crs=')
