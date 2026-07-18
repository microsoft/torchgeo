# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""STAC GeoParquet dataset support."""

import ntpath
import os
import posixpath
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path as LocalPath
from pathlib import PureWindowsPath
from typing import Any, cast
from urllib.parse import urlparse, urlunparse

import geopandas as gpd
import pandas as pd
import rasterio
import torch
from geopandas import GeoDataFrame
from pyproj import CRS

from .errors import DependencyNotFoundError
from .geo import RasterDataset
from .utils import GeoSlice, Path, Sample

DEFAULT_MAX_INDEX_ITEMS = 10_000


class STACDataset(RasterDataset):
    """STAC GeoParquet-backed raster dataset."""

    is_image = True

    def __init__(
        self,
        index_path: Path,
        asset_keys: Sequence[str],
        *,
        intersects_bbox: tuple[float, float, float, float] | None = None,
        time_range: tuple[str | pd.Timestamp, str | pd.Timestamp] | None = None,
        filters: list[tuple[str, str, object]]
        | list[list[tuple[str, str, object]]]
        | None = None,
        storage_options: Mapping[str, object] | None = None,
        sign_href: Callable[[str], str] | None = None,
        crs: CRS | None = None,
        res: float | tuple[float, float] | None = None,
        transforms: Callable[[Sample], Sample] | None = None,
        cache: bool = True,
        time_series: bool = False,
        max_index_items: int | None = DEFAULT_MAX_INDEX_ITEMS,
    ) -> None:
        """Initialize a STAC GeoParquet dataset.

        Args:
            index_path: Path or URI to a STAC GeoParquet file.
            asset_keys: STAC asset keys to read (e.g. ``('B04', 'B08')``).
            intersects_bbox: Spatial pre-filter in the GeoParquet geometry CRS.
            time_range: Inclusive UTC time filter ``(start, stop)``.
            filters: PyArrow-style filter tuples.
            storage_options: Filesystem options or HTTP headers for remote files.
            sign_href: Callback to sign asset hrefs just before opening them.
            crs: Sampling CRS. If not provided, it is inferred by opening the
                first selected raster asset.
            res: Sampling resolution. If not provided, it is inferred by opening
                the first selected raster asset.
            transforms: Optional sample transform applied in ``__getitem__``.
            cache: Cache file handles between reads to speed up repeated sampling.
            time_series: Stack intersecting items along time instead of merging.
            max_index_items: Hard cap on in-memory index size after filtering.
        """
        if not asset_keys:
            raise ValueError('asset_keys cannot be empty')
        if max_index_items is not None and max_index_items < 1:
            raise ValueError('max_index_items must be positive or None.')

        self.paths = index_path
        self.asset_keys = tuple(asset_keys)
        self.bands = self.asset_keys
        self.band_indexes = None
        self.transforms = transforms
        self.cache = cache
        self.time_series = time_series
        self._sign_href = sign_href

        # read parquet file and apply pushdown filters
        uri = str(index_path)
        scheme = urlparse(uri).scheme.lower()
        parquet_filters = filters
        parquet_bbox = intersects_bbox
        parquet_filesystem: Any | None = None

        time_filters = None
        if time_range:
            t_start, t_end = pd.Timestamp(time_range[0]), pd.Timestamp(time_range[1])
            time_filters = [('datetime', '>=', t_start), ('datetime', '<=', t_end)]
            if parquet_filters is None:
                parquet_filters = time_filters
            elif len(parquet_filters) > 0:
                base_filters = cast(list[Any], parquet_filters)
                if isinstance(base_filters[0], tuple):
                    parquet_filters = base_filters + time_filters
                elif isinstance(base_filters[0], list):
                    parquet_filters = [f + time_filters for f in base_filters]

        read_path = index_path
        if scheme:
            try:
                import fsspec

                opts = dict(storage_options or {})
                if scheme in {'http', 'https'} and opts and 'headers' not in opts:
                    opts = {'headers': opts}

                # fsspec raises ImportError when a scheme-specific backend like
                # s3fs, gcsfs, or adlfs is not installed.
                fs, read_path = fsspec.core.url_to_fs(uri, **opts)
                parquet_filesystem = fs
            except ImportError:
                pkg = {
                    's3': 's3fs',
                    'gs': 'gcsfs',
                    'gcs': 'gcsfs',
                    'abfs': 'adlfs',
                    'abfss': 'adlfs',
                }.get(scheme, 'fsspec')
                raise DependencyNotFoundError(pkg)

        def read_index() -> GeoDataFrame:
            if parquet_filters is None:
                return gpd.read_parquet(
                    read_path, bbox=parquet_bbox, filesystem=parquet_filesystem
                )
            return gpd.read_parquet(
                read_path,
                bbox=parquet_bbox,
                filters=parquet_filters,
                filesystem=parquet_filesystem,
            )

        try:
            df = read_index()
        except Exception as e:
            err_str = str(e)
            err_type = type(e).__name__

            if isinstance(e, ValueError) and 'bbox' in err_str and intersects_bbox:
                parquet_bbox = None
                df = read_index()
                minx, miny, maxx, maxy = intersects_bbox
                df = df.cx[minx:maxx, miny:maxy]
            elif time_filters and (
                'datetime' in err_str or 'timestamp' in err_str or 'Arrow' in err_type
            ):
                # GeoPandas/PyArrow vary in how they fail timestamp pushdown,
                # so retry without time filters and apply the exact filter below.
                parquet_filters = filters
                df = read_index()
            elif isinstance(e, ValueError) and 'Missing geo metadata' in err_str:
                raise ValueError('STAC GeoParquet is missing geometry.')
            else:
                raise

        # validate required columns
        cols = set(df.columns)
        if 'geometry' not in cols:
            raise ValueError('STAC GeoParquet is missing geometry.')
        if 'assets' not in cols:
            raise ValueError('STAC GeoParquet is missing required assets column.')
        if 'datetime' not in cols and not {'start_datetime', 'end_datetime'}.issubset(
            cols
        ):
            raise ValueError(
                'STAC GeoParquet must include datetime or start_datetime/end_datetime.'
            )

        # normalize datetime columns to a single start/end pair, filling missing values
        has_interval = {'start_datetime', 'end_datetime'}.issubset(cols)
        dt_col = (
            pd.to_datetime(df['datetime'], utc=True, format='ISO8601', errors='coerce')
            if 'datetime' in cols
            else None
        )

        if has_interval:
            start = pd.to_datetime(
                df['start_datetime'], utc=True, format='ISO8601', errors='coerce'
            )
            end = pd.to_datetime(
                df['end_datetime'], utc=True, format='ISO8601', errors='coerce'
            )
            if dt_col is not None:
                start = start.fillna(dt_col)
                end = end.fillna(dt_col)
        else:
            # We checked above that if the interval columns are missing,
            # 'datetime' must be present — so here dt_col is always a real
            # Series, never None.
            assert dt_col is not None
            start = end = dt_col

        if start.isna().any() or end.isna().any():
            if not has_interval:
                raise ValueError('invalid values in datetime column')
            else:
                raise ValueError('invalid values in start_datetime/end_datetime column')

        if time_range:
            t_start = (
                pd.Timestamp(time_range[0]).tz_localize('UTC')
                if pd.Timestamp(time_range[0]).tz is None
                else pd.Timestamp(time_range[0]).tz_convert('UTC')
            )
            t_end = (
                pd.Timestamp(time_range[1]).tz_localize('UTC')
                if pd.Timestamp(time_range[1]).tz is None
                else pd.Timestamp(time_range[1]).tz_convert('UTC')
            )
            if t_end < t_start:
                raise ValueError(
                    'time_range stop must be greater than or equal to start.'
                )

            time_mask = (start <= t_end) & (end >= t_start)
            df, start, end = df[time_mask], start[time_mask], end[time_mask]

        if df.empty:
            raise ValueError('No STAC items matched the requested filters.')

        if max_index_items is not None and len(df) > max_index_items:
            raise ValueError(
                f'STAC filters matched more than {len(df)} items, exceeding max_index_items={max_index_items}. Narrow the filters or increase max_index_items.'
            )

        # extract and validate asset hrefs for all requested keys
        hrefs = {}
        valid_rows = pd.Series(True, index=df.index)
        for key in self.asset_keys:
            col = df['assets'].str.get(key).str.get('href')
            hrefs[key] = col
            valid_rows &= col.fillna('').str.strip() != ''

        dropped = int((~valid_rows).sum())
        if dropped == len(df):
            raise ValueError(
                f'All STAC items were dropped because requested asset hrefs are missing: {", ".join(self.asset_keys)}.'
            )
        if dropped > 0:
            warnings.warn(
                f'Dropping {dropped} STAC item(s) missing requested asset hrefs.',
                UserWarning,
            )

        df, start, end = (
            df[valid_rows].reset_index(drop=True),
            start[valid_rows].reset_index(drop=True),
            end[valid_rows].reset_index(drop=True),
        )
        for key in self.asset_keys:
            hrefs[key] = (
                hrefs[key][valid_rows]
                .reset_index(drop=True)
                .apply(lambda h: self._resolve_href(str(h), index_path))
            )

        # sort items chronologically and by asset hrefs to ensure deterministic order
        sort_df = pd.DataFrame({'start': start, 'end': end, **hrefs})
        order = sort_df.sort_values(
            list(sort_df.columns), kind='mergesort'
        ).index.to_numpy()

        df = df.iloc[order].reset_index(drop=True)
        start = start.iloc[order].reset_index(drop=True)
        end = end.iloc[order].reset_index(drop=True)
        hrefs = {k: v.iloc[order].reset_index(drop=True) for k, v in hrefs.items()}

        # infer sampling crs and resolution from the first asset if not provided
        if crs is None or res is None:
            first_href = str(hrefs[self.asset_keys[0]].iloc[0])

            with self._load_warp_file(first_href) as src:
                inferred_crs = src.crs
                inferred_res = src.res

        sampling_crs = crs if crs is not None else inferred_crs

        if res is not None:
            if isinstance(res, (int, float)):
                res_tup = (float(res), float(res))
            else:
                res_tup = (float(res[0]), float(res[1]))
            if res_tup[0] <= 0 or res_tup[1] <= 0:
                raise ValueError('res values must be positive.')
            self._res = res_tup
        else:
            self._res = inferred_res

        # populate the dataset index for sampling
        if df.crs is None:
            warnings.warn(
                'GeoParquet geometry CRS metadata is missing; defaulting to EPSG:4326.',
                UserWarning,
            )
            df = df.set_crs(CRS.from_epsg(4326), allow_override=True)
        geometry = df.geometry
        if sampling_crs != df.crs:
            geometry = geometry.to_crs(sampling_crs)

        self.index = GeoDataFrame(
            {key: hrefs[key].to_list() for key in self.asset_keys},
            index=pd.IntervalIndex.from_arrays(
                start, end, closed='both', name='datetime'
            ),
            geometry=geometry.to_numpy(),
            crs=sampling_crs,
        )

    @property
    def files(self) -> list[str]:
        """Resolved asset hrefs in index order, deduplicated."""
        hrefs = [
            str(href)
            for row in zip(*(self.index[key] for key in self.asset_keys))
            for href in row
        ]
        return list(dict.fromkeys(hrefs))

    def __getitem__(self, index: GeoSlice) -> Sample:
        """Return a raster sample for the given spatiotemporal slice."""
        x, y, t = self._disambiguate_slice(index)
        interval = pd.Interval(t.start, t.stop, closed='both')
        matches = self.index.iloc[self.index.index.overlaps(interval)][:: t.step]
        matches = matches.cx[x.start : x.stop, y.start : y.stop]
        if matches.empty:
            raise IndexError(
                f'index: {index} not found in dataset with bounds: {self.bounds}'
            )

        bands = [
            self._merge_or_stack(matches[key], index, self.band_indexes)
            for key in self.asset_keys
        ]
        sample: Sample = {
            'bounds': self._slice_to_tensor(index),
            'image': torch.cat(bands, dim=-3).to(self.dtype),
            'transform': torch.tensor(
                rasterio.transform.from_origin(x.start, y.stop, x.step, y.step)
            ),
        }
        if self.transforms is not None:
            sample = self.transforms(sample)
        return sample

    def _load_warp_file(
        self, filepath: Path, crs: CRS | None = None
    ) -> rasterio.io.DatasetReader:
        """Sign href just before opening so the cache key stays canonical."""
        if self._sign_href is not None:
            filepath = self._sign_href(str(filepath))
        return super()._load_warp_file(filepath, crs)

    @staticmethod
    def _resolve_href(href: str, index_path: object) -> str:
        """Resolve a relative asset href against the GeoParquet file location."""
        if (
            href.startswith('/')
            or (len(href) >= 3 and href[1] == ':' and href[0].isalpha())
            or href.startswith('\\\\')
            or urlparse(href).scheme
        ):
            return href
        index = str(index_path)
        parsed = urlparse(index)
        is_windows = (
            len(index) >= 3 and index[1] == ':' and index[0].isalpha()
        ) or index.startswith('\\\\')
        if parsed.scheme and not is_windows:
            resolved = posixpath.normpath(
                posixpath.join(posixpath.dirname(parsed.path), href)
            )
            return urlunparse(parsed._replace(path=resolved))
        if is_windows:
            base = PureWindowsPath(index)
            if base.suffix:
                base = base.parent
            return ntpath.normpath(str(base / PureWindowsPath(href)))
        base_path = LocalPath(index)
        base_dir = base_path if base_path.is_dir() else base_path.parent
        return os.path.normpath(base_dir / href)
