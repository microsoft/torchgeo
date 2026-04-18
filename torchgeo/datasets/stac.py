# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""STAC GeoParquet-backed datasets.

This module implements the runtime/dataset-index layer for STAC support: a
:class:`GeoDataset` that reads a materialized STAC GeoParquet file into a
:class:`geopandas.GeoDataFrame` index and reuses the existing
:class:`RasterDataset` read path (rasterio + WarpedVRT) at sample time.

The materialization step (querying a STAC API and writing the GeoParquet) is
intentionally out of scope here. It is a separate concern that introduces
latency, rate limiting, auth/signing, and provider-specific quirks into the
dataloader path. Keeping discovery offline and sampling local lines up with
the existing ``RasterDataset`` mental model.
"""

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import ClassVar

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import torch
from matplotlib.figure import Figure
from pyproj import CRS
from torch import Tensor

from .errors import DatasetNotFoundError, RGBBandsMissingError
from .geo import RasterDataset
from .utils import GeoSlice, Path, Sample


class STACDataset(RasterDataset):
    """Base class for STAC GeoParquet-backed raster datasets.

    A :class:`STACDataset` builds its spatiotemporal index from a STAC
    GeoParquet file rather than by walking a directory of files. The parquet
    must follow (a subset of) the
    `STAC GeoParquet <https://github.com/stac-utils/stac-geoparquet>`_ schema:

    * a ``geometry`` column (GeoParquet standard)
    * a ``datetime`` column (or both ``start_datetime`` and ``end_datetime``)
    * either an ``assets`` struct column whose fields are asset names mapping
      to objects with an ``href`` field, OR one column per asset name whose
      values are the asset HREFs directly.

    The configured *bands* are interpreted as STAC asset names. Each asset is
    treated as a separate file, mirroring ``RasterDataset(separate_files=True)``.

    Subclasses are encouraged to bake provider/collection semantics into class
    attributes (default ``all_bands``, ``dtype``, ``is_image``, etc.) following
    the same pattern as the rest of the library, instead of moving those
    quirks into user code.

    .. versionadded:: 0.10
    """

    #: Asset name → human-readable description (optional, used for plotting).
    asset_descriptions: ClassVar[dict[str, str]] = {}

    #: Default RGB asset names for :meth:`plot`. Override in subclasses.
    rgb_bands: ClassVar[tuple[str, str, str] | None] = None

    # All STAC datasets stack per-asset files at sample time.
    separate_files = True

    def __init__(
        self,
        paths: Path | Iterable[Path] = 'data',
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        bands: Sequence[str] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        time_series: bool = False,
        asset_columns: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize a new STACDataset instance.

        Args:
            paths: path or URL to a STAC GeoParquet file (or an iterable of
                them; multiple files are concatenated).
            crs: :term:`coordinate reference system (CRS)` to warp to
                (defaults to the CRS of the parquet's geometry column).
            res: resolution of the dataset in units of CRS (defaults to the
                resolution of the first asset opened at sample time).
            bands: STAC asset names to load. Defaults to :attr:`all_bands`.
            transforms: a function/transform that takes an input sample and
                returns a transformed version.
            cache: if True, cache file handles to speed up repeated sampling.
            time_series: if True, stack data along the time-series dimension
                ``[T, C, H, W]`` instead of mosaicking.
            asset_columns: optional mapping of asset name → parquet column
                name, used when the column names do not match the asset names
                directly. Ignored if the parquet has a STAC ``assets`` struct
                column.

        Raises:
            DatasetNotFoundError: if no rows are found in the parquet.
            AssertionError: if requested bands are not in :attr:`all_bands`.
        """
        # NOTE: we deliberately do not call RasterDataset.__init__ -- it would
        # walk the filesystem and open every file with rasterio. Instead we
        # build self.index directly from the parquet.
        self.paths = paths
        self.bands = bands or self.all_bands
        self.transforms = transforms
        self.cache = cache
        self.time_series = time_series
        self.asset_columns = dict(asset_columns) if asset_columns else {}

        if self.all_bands:
            assert set(self.bands) <= set(self.all_bands), (
                f'requested bands {set(self.bands) - set(self.all_bands)} are '
                f'not in {type(self).__name__}.all_bands'
            )

        df = self._read_parquet(paths)
        df = self._extract_asset_hrefs(df)
        self.index = self._build_index(df, crs)

        # Set the band-indexes attribute RasterDataset expects. For per-asset
        # files we always read band 1.
        self.band_indexes = None

        if res is None:
            # Peek at the first asset to learn the native resolution. This is
            # one open instead of one-per-file like RasterDataset.
            href = self.index[f'href_{self.bands[0]}'].iloc[0]
            with rasterio.open(href) as src:
                res = src.res
        if isinstance(res, int | float):
            res = (res, res)
        self._res = res

    # ------------------------------------------------------------------ #
    # Index construction
    # ------------------------------------------------------------------ #

    def _read_parquet(self, paths: Path | Iterable[Path]) -> gpd.GeoDataFrame:
        """Read one or more STAC GeoParquet files into a GeoDataFrame.

        Args:
            paths: path/URL or iterable of paths/URLs.

        Returns:
            Concatenated GeoDataFrame.

        Raises:
            DatasetNotFoundError: if the result is empty.
        """
        if isinstance(paths, str | os.PathLike):
            path_list: list[Path] = [paths]
        else:
            path_list = list(paths)

        frames = [gpd.read_parquet(p) for p in path_list]
        if not frames:
            raise DatasetNotFoundError(self)

        df = pd.concat(frames, ignore_index=True)
        if df.empty:
            raise DatasetNotFoundError(self)

        # pd.concat returns a plain DataFrame; restore as GeoDataFrame.
        if not isinstance(df, gpd.GeoDataFrame):
            df = gpd.GeoDataFrame(df, geometry='geometry', crs=frames[0].crs)
        return df

    def _extract_asset_hrefs(self, df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Resolve each requested band to an HREF column on *df*.

        Supports two parquet shapes:

        * STAC GeoParquet native: a single ``assets`` struct column whose
          fields map asset name → ``{href, type, ...}``.
        * Pre-flattened: one column per asset name (or per
          :attr:`asset_columns` mapping) holding HREFs directly.

        Args:
            df: GeoDataFrame from the parquet.

        Returns:
            The same GeoDataFrame with an ``href_<band>`` column for each
            band in :attr:`bands`.

        Raises:
            KeyError: if an asset is not found in the parquet.
        """
        if 'assets' in df.columns:
            for band in self.bands:
                df[f'href_{band}'] = df['assets'].apply(
                    lambda a, b=band: a[b]['href']
                    if isinstance(a, Mapping) and b in a
                    else None
                )
        else:
            for band in self.bands:
                col = self.asset_columns.get(band, band)
                if col not in df.columns:
                    raise KeyError(
                        f"asset '{band}' not found in parquet "
                        f"(looked for column '{col}'). Pass asset_columns to "
                        'override the column mapping.'
                    )
                df[f'href_{band}'] = df[col]

        # Drop rows missing any requested asset.
        href_cols = [f'href_{b}' for b in self.bands]
        df = df.dropna(subset=href_cols)
        if df.empty:
            raise DatasetNotFoundError(self)
        return df

    def _build_index(
        self, df: gpd.GeoDataFrame, crs: CRS | None
    ) -> gpd.GeoDataFrame:
        """Build the GeoDataFrame index expected by :class:`GeoDataset`.

        Args:
            df: GeoDataFrame with at least ``geometry`` and one ``href_<band>``
                column per requested band, plus ``datetime`` (or
                ``start_datetime``/``end_datetime``).
            crs: target CRS to reproject geometries into.

        Returns:
            GeoDataFrame indexed by a ``pd.IntervalIndex`` over time, with one
            ``href_<band>`` column per band and a ``filepath`` column equal to
            ``href_<bands[0]>`` so that base-class code reading ``df.filepath``
            still works.
        """
        # Datetime extraction: prefer (start_datetime, end_datetime) if both
        # present, else fall back to a single datetime column.
        if {'start_datetime', 'end_datetime'} <= set(df.columns):
            mints = pd.to_datetime(df['start_datetime'], utc=True).dt.tz_localize(None)
            maxts = pd.to_datetime(df['end_datetime'], utc=True).dt.tz_localize(None)
        elif 'datetime' in df.columns:
            t = pd.to_datetime(df['datetime'], utc=True).dt.tz_localize(None)
            mints = maxts = t
        else:
            mints = pd.Series([self.mint] * len(df))
            maxts = pd.Series([self.maxt] * len(df))

        time_index = pd.IntervalIndex.from_arrays(
            mints, maxts, closed='both', name='datetime'
        )

        # Reproject geometries if requested.
        if crs is not None and df.crs is not None and CRS(df.crs) != CRS(crs):
            df = df.to_crs(crs)
        elif crs is not None and df.crs is None:
            df = df.set_crs(crs)

        href_cols = [f'href_{b}' for b in self.bands]
        data = {col: df[col].to_numpy() for col in href_cols}
        # Provide a `filepath` column for compatibility with code that reads
        # df.filepath on the index (e.g. _update_filepath in RasterDataset).
        data['filepath'] = df[href_cols[0]].to_numpy()

        return gpd.GeoDataFrame(
            data, index=time_index, geometry=df.geometry.to_numpy(), crs=df.crs
        )

    # ------------------------------------------------------------------ #
    # Sampling
    # ------------------------------------------------------------------ #

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Retrieve image data for a spatiotemporal slice.

        Each requested band is read from its asset HREF and either mosaicked
        (``time_series=False``) or stacked (``time_series=True``) across the
        items intersecting the slice.

        Args:
            index: ``[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]``
                coordinates to index.

        Returns:
            Sample dict with ``image`` (or ``mask``), ``bounds``, ``transform``.

        Raises:
            IndexError: if no items intersect the slice.
        """
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop)
        df = self.index.iloc[self.index.index.overlaps(interval)]
        df = df.iloc[:: t.step]
        df = df.cx[x.start : x.stop, y.start : y.stop]

        if df.empty:
            raise IndexError(
                f'index: {index} not found in dataset with bounds: {self.bounds}'
            )

        # One per-band tensor, then concat along channel dim — same shape
        # contract as RasterDataset(separate_files=True).
        data_list: list[Tensor] = []
        for band in self.bands:
            band_hrefs = list(df[f'href_{band}'])
            data_list.append(self._merge_or_stack(band_hrefs, index))
        data = torch.cat(data_list, dim=-3)

        transform = rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
        sample: Sample = {
            'bounds': self._slice_to_tensor(index),
            'transform': torch.tensor(transform),
        }

        data = data.to(self.dtype)
        if self.is_image:
            sample['image'] = data
        else:
            sample['mask'] = data.squeeze(-3)

        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    # ------------------------------------------------------------------ #
    # Plotting (subclasses override)
    # ------------------------------------------------------------------ #

    def plot(
        self,
        sample: Sample,
        show_titles: bool = True,
        suptitle: str | None = None,
    ) -> Figure:
        """Plot a sample, using :attr:`rgb_bands` if set.

        Args:
            sample: a sample returned by :meth:`__getitem__`.
            show_titles: whether to draw a panel title.
            suptitle: optional supertitle for the figure.

        Returns:
            A matplotlib :class:`~matplotlib.figure.Figure`.

        Raises:
            RGBBandsMissingError: if :attr:`rgb_bands` is not set or not loaded.
        """
        if self.rgb_bands is None or not set(self.rgb_bands).issubset(self.bands):
            raise RGBBandsMissingError()

        rgb_indexes = [self.bands.index(b) for b in self.rgb_bands]
        image = sample['image'][rgb_indexes].permute(1, 2, 0).float()
        # Robust 2-98 percentile contrast stretch — STAC assets vary wildly.
        lo, hi = np.percentile(image.numpy(), [2, 98])
        if hi > lo:
            image = ((image - lo) / (hi - lo)).clamp(0, 1)

        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(image.numpy())
        ax.axis('off')
        if show_titles:
            ax.set_title('Image')
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig


class Sentinel2STAC(STACDataset):
    """Sentinel-2 L2A read from a STAC GeoParquet index.

    This is a thin profile over :class:`STACDataset` that bakes in the
    Sentinel-2 L2A band list and a sensible default RGB triplet. It is
    intentionally provider-agnostic: point ``paths`` at any STAC GeoParquet
    that publishes Sentinel-2 L2A items (e.g. one materialized from
    Microsoft Planetary Computer, Element-84 Earth Search, or CDSE).

    .. versionadded:: 0.10
    """

    all_bands: tuple[str, ...] = (
        'B01',
        'B02',
        'B03',
        'B04',
        'B05',
        'B06',
        'B07',
        'B08',
        'B8A',
        'B09',
        'B11',
        'B12',
    )

    rgb_bands = ('B04', 'B03', 'B02')
