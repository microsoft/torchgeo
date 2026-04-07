# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Rasteret-backed RasterDataset.

This is a collection-first experimental backend for raster-style TorchGeo
workflows. TorchGeo exposes a first-class ``RasteretDataset`` entry point,
while Rasteret keeps the collection preparation and accelerated COG I/O in its
own public API.

.. warning::
   This dataset is considered experimental and subject to change.

.. versionadded:: 0.10
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import torch
from pyproj import CRS

from torchgeo.datasets.geo import RasterDataset, Sample
from torchgeo.datasets.utils import GeoSlice, lazy_import


class RasteretDataset(RasterDataset):
    """A :class:`RasterDataset` that delegates reads to Rasteret.

    The class exists to expose Rasteret as a first-class TorchGeo dataset while
    keeping TorchGeo decoupled from Rasteret internals. Unlike native
    :class:`RasterDataset` subclasses, this dataset starts from a Rasteret
    ``Collection`` instead of filesystem paths. The heavy lifting stays in
    Rasteret's own adapter implementation.

    Construction-time ``crs=...`` remains supported, but published Rasteret
    ``0.3.x`` delegates do not safely support post-init CRS mutation. To change
    CRS, construct a new dataset with ``crs=...``.

    .. warning::
       This dataset is considered experimental and subject to change.

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
        cloud_config: Any = None,
        backend: Any = None,
        allow_resample: bool = False,
        label_field: str | None = None,
        geometries: Any = None,
        geometries_crs: int = 4326,
    ) -> None:
        """Initialize a new RasteretDataset instance.

        Args:
            collection: A Rasteret ``Collection`` object, typically obtained
                from ``rasteret.build(...)``, ``rasteret.build_from_stac(...)``,
                ``rasteret.build_from_table(...)``, or ``rasteret.load(...)``.
            bands: Band codes to load.
            crs: CRS to warp to.
            res: Output resolution in units of CRS.
            transforms: A function/transform that takes an input sample and
                returns a transformed version.
            cache: Retained for API compatibility with ``RasterDataset``.
            time_series: If ``True``, stack data along T dimension.
            is_image: If ``True``, return values in ``sample['image']``.
            max_concurrent: Maximum concurrent HTTP byte-range requests.
            cloud_config: Rasteret ``CloudConfig`` for authenticated reads.
            backend: Rasteret ``StorageBackend`` instance.
            allow_resample: Allow Rasteret to resample bands with mixed native
                resolutions.
            label_field: Collection column name to expose as ``sample['label']``.
            geometries: Optional geometry filter applied before sampling.
            geometries_crs: EPSG code for *geometries*.

        Raises:
            ValueError: If no bands are provided or *crs* is not EPSG-resolvable.
            TypeError: If *collection* does not expose
                ``to_torchgeo_dataset(...)``.
        """
        lazy_import('rasteret')

        if not bands:
            raise ValueError('At least one band is required')
        to_torchgeo_dataset = getattr(collection, 'to_torchgeo_dataset', None)
        if not callable(to_torchgeo_dataset):
            raise TypeError(
                'collection must be a rasteret.Collection with to_torchgeo_dataset(...)'
            )

        target_crs: int | None = None
        if crs is not None:
            epsg = crs.to_epsg()
            if epsg is None:
                raise ValueError('RasteretDataset requires an EPSG CRS')
            target_crs = int(epsg)

        self._delegate = to_torchgeo_dataset(
            bands=list(bands),
            is_image=is_image,
            allow_resample=allow_resample,
            label_field=label_field,
            geometries=geometries,
            geometries_crs=geometries_crs,
            transforms=transforms,
            max_concurrent=max_concurrent,
            cloud_config=cloud_config,
            backend=backend,
            time_series=time_series,
            target_crs=target_crs,
        )

        # RasterDataset-compatible attributes.
        self.paths = ''
        self.bands = tuple(bands)
        self.all_bands = tuple(bands)
        self.transforms = transforms
        self.cache = cache
        self.time_series = time_series
        self.is_image = is_image
        self.separate_files = False
        self.band_indexes = None
        self.index = self._delegate.index
        self._res = self._delegate.res

        if res is not None:
            self.res = res

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve a sample indexed by spatiotemporal slice.

        Args:
            index: ``[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]`` query.

        Returns:
            Sample at the requested index.
        """
        return self._delegate[index]

    @property
    def crs(self) -> CRS:
        """Coordinate reference system of the dataset.

        Returns:
            The dataset CRS.
        """
        return self._delegate.crs

    @crs.setter
    def crs(self, new_crs: CRS) -> None:
        """Reject post-init CRS changes.

        Args:
            new_crs: New CRS.

        Raises:
            AttributeError: Always raised. Rasteret binds read-time CRS at
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
        return self._delegate.res

    @res.setter
    def res(self, new_res: float | tuple[float, float]) -> None:
        """Change dataset resolution.

        Args:
            new_res: New resolution in ``(xres, yres)`` format.
        """
        self._delegate.res = new_res
        self._res = self._delegate.res

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

    def close(self) -> None:
        """Close Rasteret background resources if supported by delegate."""
        close = getattr(self._delegate, 'close', None)
        if callable(close):
            close()
