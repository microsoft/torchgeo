# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import copy
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, cast

import geopandas as gpd
import pandas as pd
import pytest
import torch
from geopandas import GeoDataFrame
from pyproj import CRS

from torchgeo.datasets import STACDataset
from torchgeo.datasets.errors import DependencyNotFoundError

pytest.importorskip('pyarrow')
DATA = Path(__file__).parents[1] / 'data' / 'stac'
ITEMS = DATA / 'items.parquet'


def _raise(exc: BaseException) -> NoReturn:
    """Raise ``exc`` — lets a monkeypatched callable fail from inside a lambda."""
    raise exc


def _editable_table() -> GeoDataFrame:
    """Load the committed item table with asset hrefs made absolute.

    Tests that need a schema variant tweak this table and write it to a
    temporary path; absolute hrefs keep assets pointing at committed fixtures.
    """
    table = gpd.read_parquet(ITEMS)
    for idx in table.index:
        assets = copy.deepcopy(table.at[idx, 'assets'])
        for asset in assets.values():
            asset['href'] = str(DATA / cast(str, asset['href']))
        table.at[idx, 'assets'] = assets
    return table


def _sample_query(dataset: STACDataset) -> tuple[slice, slice, slice]:
    """A small spatiotemporal slice centered on the first indexed item."""
    point = dataset.index.geometry.iloc[0].representative_point()
    xres, yres = dataset.res
    _, _, time = dataset.bounds
    return (
        slice(point.x - xres * 8, point.x + xres * 8, xres),
        slice(point.y - yres * 8, point.y + yres * 8, yres),
        slice(time.start, time.stop, 1),
    )


def _inner_bbox(table: GeoDataFrame) -> tuple[float, float, float, float]:
    """A bbox covering only the center quarter of the first item's geometry."""
    minx, miny, maxx, maxy = table.geometry.iloc[0].bounds
    dx, dy = ((maxx - minx) * 0.25, (maxy - miny) * 0.25)
    return (minx + dx, miny + dy, maxx - dx, maxy - dy)


class TestSTACDataset:
    """Behavior of the dataset, driven through its public API."""

    def test_reads_a_multiband_sample(self) -> None:
        dataset = STACDataset(ITEMS, ('B04', 'B08'))
        sample = dataset[_sample_query(dataset)]
        assert len(dataset) == 4
        assert dataset.crs.to_epsg() == 32632
        assert dataset.res == (10.0, 10.0)
        assert sample['image'].shape == (2, 16, 16)
        torch.testing.assert_close(sample['image'][0], torch.ones(16, 16))
        torch.testing.assert_close(sample['image'][1], torch.full((16, 16), 2.0))

    def test_files_lists_resolved_unsigned_hrefs(self) -> None:
        dataset = STACDataset(ITEMS, ('B04', 'B08'))
        assert dataset.files == [
            str(DATA / 'rasters' / f'item-{i}_{b}.tif')
            for i in range(4)
            for b in ('B04', 'B08')
        ]

    def test_reads_a_time_series_sample(self) -> None:
        dataset = STACDataset(ITEMS, ('B04',), time_series=True)
        sample = dataset[_sample_query(dataset)]
        assert sample['image'].shape == (1, 1, 16, 16)
        torch.testing.assert_close(sample['image'][0, 0], torch.ones(16, 16))

    def test_stacks_overlapping_items_along_time(self, tmp_path: Path) -> None:
        source = _editable_table()
        table = source.iloc[[1, 0]].copy().reset_index(drop=True)
        for idx, href in enumerate(('item-0_B08.tif', 'item-0_B04.tif')):
            assets = cast(dict[str, Any], copy.deepcopy(source.at[0, 'assets']))
            assets['B04']['href'] = str(DATA / 'rasters' / href)
            table.at[idx, 'assets'] = cast(Any, assets)
            table.at[idx, 'geometry'] = source.geometry.iloc[0]
        table.to_parquet(tmp_path / 'items.parquet')
        dataset = STACDataset(tmp_path / 'items.parquet', ('B04',), time_series=True)
        sample = dataset[_sample_query(dataset)]
        assert sample['image'].shape == (2, 1, 16, 16)
        assert [float(image[0, 0, 0]) for image in sample['image']] == [1.0, 2.0]
        x, y, t = _sample_query(dataset)
        stepped = dataset[x, y, slice(t.start, t.stop, 2)]
        assert stepped['image'].shape == (1, 1, 16, 16)
        torch.testing.assert_close(stepped['image'][0, 0], torch.ones(16, 16))

    def test_asset_keys_may_contain_periods(self, tmp_path: Path) -> None:
        table = _editable_table()
        for idx in table.index:
            assets = cast(dict[str, Any], table.at[idx, 'assets'])
            assets['mtl.json'] = assets.pop('B04')
        table.to_parquet(tmp_path / 'items.parquet')
        dataset = STACDataset(tmp_path / 'items.parquet', ('mtl.json',))
        assert len(dataset) == 4
        assert dataset[_sample_query(dataset)]['image'].shape == (1, 16, 16)

    def test_bbox_selects_items_it_does_not_clip_pixels(self) -> None:
        table = gpd.read_parquet(ITEMS)
        dataset = STACDataset(ITEMS, ('B04',), intersects_bbox=_inner_bbox(table))
        assert len(dataset) == 1

    def test_bbox_and_property_filter_combine(self) -> None:
        table = gpd.read_parquet(ITEMS)
        dataset = STACDataset(
            ITEMS,
            ('B04',),
            intersects_bbox=_inner_bbox(table),
            filters=[('eo:cloud_cover', '<', 10.0)],
        )
        assert len(dataset) == 1

    def test_bbox_without_a_covering_bbox_column(self, tmp_path: Path) -> None:
        table = _editable_table()
        table.to_parquet(tmp_path / 'items.parquet', write_covering_bbox=False)
        dataset = STACDataset(
            tmp_path / 'items.parquet', ('B04',), intersects_bbox=_inner_bbox(table)
        )
        assert len(dataset) == 1

    def test_explicit_crs_and_res_are_honored(self) -> None:
        dataset = STACDataset(ITEMS, ('B04',), crs=CRS.from_epsg(3857), res=12.0)
        assert dataset.crs.to_epsg() == 3857
        assert dataset.res == (12.0, 12.0)

    @pytest.mark.parametrize('schema', ['datetime', 'naive', 'interval', 'nullable'])
    def test_time_range_selects_items(self, tmp_path: Path, schema: str) -> None:
        table = _editable_table()
        if schema == 'naive':
            table['datetime'] = pd.to_datetime(table['datetime']).dt.tz_localize(None)
        elif schema in {'interval', 'nullable'}:
            times = pd.to_datetime(table['datetime'])
            table['start_datetime'] = times - pd.Timedelta(minutes=2)
            table['end_datetime'] = times + pd.Timedelta(minutes=2)
            table['datetime'] = pd.NaT if schema == 'nullable' else table['datetime']
            if schema == 'interval':
                table = table.drop(columns=['datetime'])
        table.to_parquet(tmp_path / 'items.parquet')
        dataset = STACDataset(
            tmp_path / 'items.parquet',
            ('B04',),
            time_range=('2020-01-01T00:00:00', '2020-01-01T23:59:59Z'),
        )
        assert len(dataset) == 2

    def test_time_range_with_both_datetime_and_interval_columns(
        self, tmp_path: Path
    ) -> None:
        table = _editable_table()
        table['start_datetime'] = pd.NaT
        table['end_datetime'] = pd.NaT
        table.to_parquet(tmp_path / 'items.parquet')
        dataset = STACDataset(
            tmp_path / 'items.parquet',
            ('B04',),
            time_range=('2020-01-01T00:00:00Z', '2020-01-02T23:59:59Z'),
        )
        assert len(dataset) == 4

    def test_time_range_extends_existing_filters(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table = _editable_table()
        calls: list[object] = []

        def read_parquet(*args: object, **kwargs: object) -> GeoDataFrame:
            calls.append(kwargs.get('filters'))
            return table.copy()

        monkeypatch.setattr(gpd, 'read_parquet', read_parquet)
        STACDataset(
            ITEMS,
            ('B04',),
            filters=[('eo:cloud_cover', '<', 20.0)],
            time_range=('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z'),
            crs=CRS.from_epsg(32632),
            res=10,
        )
        STACDataset(
            ITEMS,
            ('B04',),
            filters=[[('eo:cloud_cover', '<', 10.0)]],
            time_range=('2020-01-01T00:00:00Z', '2020-01-02T00:00:00Z'),
            crs=CRS.from_epsg(32632),
            res=10,
        )

        assert calls[0] == [
            ('eo:cloud_cover', '<', 20.0),
            ('datetime', '>=', pd.Timestamp('2020-01-01T00:00:00Z')),
            ('datetime', '<=', pd.Timestamp('2020-01-02T00:00:00Z')),
        ]
        assert calls[1] == [
            [
                ('eo:cloud_cover', '<', 10.0),
                ('datetime', '>=', pd.Timestamp('2020-01-01T00:00:00Z')),
                ('datetime', '<=', pd.Timestamp('2020-01-02T00:00:00Z')),
            ]
        ]

    @pytest.mark.parametrize(
        ('filters', 'expected'),
        [
            ([('eo:cloud_cover', '==', 5.0)], 1),
            ([('eo:cloud_cover', '<', 20.0)], 2),
            ([('eo:cloud_cover', '>=', 25.0)], 2),
            ([('eo:cloud_cover', '!=', 5.0)], 3),
            ([('eo:cloud_cover', 'in', [5.0, 25.0])], 2),
            ([('eo:cloud_cover', 'not in', [5.0, 25.0])], 2),
        ],
    )
    def test_filters_select_items(
        self, filters: list[tuple[str, str, object]], expected: int
    ) -> None:
        assert len(STACDataset(ITEMS, ('B04',), filters=filters)) == expected

    @pytest.mark.parametrize(
        ('columns', 'message'),
        [
            ({'assets', 'datetime'}, 'missing geometry'),
            ({'geometry', 'datetime'}, 'missing required assets'),
            ({'geometry', 'assets'}, 'must include datetime'),
            ({'geometry', 'assets', 'start_datetime'}, 'must include datetime'),
        ],
    )
    def test_required_columns_are_validated(
        self, tmp_path: Path, columns: set[str], message: str
    ) -> None:
        table = _editable_table()
        for col in table.columns:
            if col not in columns and col != 'geometry':
                table = table.drop(columns=[col])
        if 'geometry' not in columns:
            table = pd.DataFrame(table.drop(columns=['geometry']))
        table.to_parquet(tmp_path / 'items.parquet')
        with pytest.raises(ValueError, match=message):
            STACDataset(tmp_path / 'items.parquet', ('B04',))

    def test_missing_geometry_column_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        table = pd.DataFrame(
            {'assets': [{}], 'datetime': [pd.Timestamp('2020-01-01T00:00:00Z')]}
        )

        def read_parquet(*args: object, **kwargs: object) -> GeoDataFrame:
            return cast(GeoDataFrame, table)

        monkeypatch.setattr(gpd, 'read_parquet', read_parquet)
        with pytest.raises(ValueError, match='missing geometry'):
            STACDataset('items.parquet', ('B04',), crs=CRS.from_epsg(32632), res=10)

    def test_missing_geo_metadata_error_is_clear(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            gpd,
            'read_parquet',
            lambda *args, **kwargs: _raise(ValueError('Missing geo metadata')),
        )
        with pytest.raises(ValueError, match='missing geometry'):
            STACDataset('items.parquet', ('B04',))

    def test_read_errors_are_not_swallowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            gpd,
            'read_parquet',
            lambda *args, **kwargs: _raise(RuntimeError('bad read')),
        )
        with pytest.raises(RuntimeError, match='bad read'):
            STACDataset('items.parquet', ('B04',))

    def test_invalid_constructor_inputs_raise(self) -> None:
        with pytest.raises(ValueError, match='asset_keys cannot be empty'):
            STACDataset(ITEMS, ())
        with pytest.raises(ValueError, match='must be positive'):
            STACDataset(ITEMS, ('B04',), max_index_items=0)
        for bad_res in (0, -10):
            with pytest.raises(ValueError, match='res values must be positive'):
                STACDataset(ITEMS, ('B04',), res=bad_res)
        with pytest.raises(ValueError, match='time_range stop'):
            STACDataset(
                ITEMS,
                ('B04',),
                time_range=('2020-01-02T00:00:00Z', '2020-01-01T00:00:00Z'),
            )
        with pytest.raises(
            ValueError, match='No STAC items matched the requested filters'
        ):
            STACDataset(
                ITEMS,
                ('B04',),
                time_range=('1999-01-01T00:00:00Z', '1999-01-02T00:00:00Z'),
            )
        with pytest.raises(ValueError, match='No STAC items matched'):
            STACDataset(ITEMS, ('B04',), filters=[('eo:cloud_cover', '<', -1)])
        with pytest.raises(ValueError, match='exceeding max_index_items=1'):
            STACDataset(ITEMS, ('B04',), max_index_items=1)
        assert STACDataset(ITEMS, ('B04',), res=(10, 20)).res == (10.0, 20.0)
        assert len(STACDataset(ITEMS, ('B04',), max_index_items=None)) == 4

    def test_remote_read_uses_fsspec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        table = _editable_table()
        calls: dict[str, object] = {}

        def url_to_fs(uri: str, **kwargs: object) -> tuple[object, str]:
            calls['uri'] = uri
            calls['kwargs'] = kwargs
            return 'fs', 'bucket/items.parquet'

        def read_parquet(path: object, **kwargs: object) -> GeoDataFrame:
            calls['path'] = path
            calls['filesystem'] = kwargs.get('filesystem')
            return table.copy()

        monkeypatch.setitem(
            sys.modules,
            'fsspec',
            SimpleNamespace(core=SimpleNamespace(url_to_fs=url_to_fs)),
        )
        monkeypatch.setattr(gpd, 'read_parquet', read_parquet)

        dataset = STACDataset(
            'https://example.com/items.parquet',
            ('B04',),
            storage_options={'Authorization': 'Bearer token'},
            crs=CRS.from_epsg(32632),
            res=10,
        )

        assert len(dataset) == 4
        assert calls == {
            'uri': 'https://example.com/items.parquet',
            'kwargs': {'headers': {'Authorization': 'Bearer token'}},
            'path': 'bucket/items.parquet',
            'filesystem': 'fs',
        }

    def test_missing_fsspec_backend_names_package(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(
            sys.modules,
            'fsspec',
            SimpleNamespace(
                core=SimpleNamespace(
                    url_to_fs=lambda *args, **kwargs: _raise(ImportError())
                )
            ),
        )
        with pytest.raises(DependencyNotFoundError, match='s3fs'):
            STACDataset(
                's3://bucket/items.parquet', ('B04',), crs=CRS.from_epsg(32632), res=10
            )

    def test_bbox_path_enforces_max_index_items(self) -> None:
        world = cast(
            tuple[float, float, float, float],
            tuple(gpd.read_parquet(ITEMS).total_bounds),
        )
        with pytest.raises(ValueError, match='exceeding max_index_items=1'):
            STACDataset(ITEMS, ('B04',), intersects_bbox=world, max_index_items=1)

    def test_bbox_outside_the_data_matches_nothing(self) -> None:
        with pytest.raises(ValueError, match='No STAC items matched'):
            STACDataset(ITEMS, ('B04',), intersects_bbox=(0.0, 0.0, 1.0, 1.0))

    @pytest.mark.parametrize('schema', ['datetime', 'interval'])
    def test_invalid_datetime_values_raise(self, tmp_path: Path, schema: str) -> None:
        table = _editable_table()
        table['datetime'] = pd.to_datetime(table['datetime']).dt.strftime(
            '%Y-%m-%dT%H:%M:%SZ'
        )
        if schema == 'datetime':
            table.at[0, 'datetime'] = 'not-a-timestamp'
            message = 'invalid values in datetime column'
        else:
            table['start_datetime'] = table['datetime']
            table['end_datetime'] = table['datetime']
            table = table.drop(columns=['datetime'])
            table.at[0, 'start_datetime'] = 'not-a-timestamp'
            message = 'invalid values in start_datetime'
        table.to_parquet(tmp_path / 'items.parquet')
        with pytest.raises(ValueError, match=message):
            STACDataset(tmp_path / 'items.parquet', ('B04',))

    def test_rows_missing_asset_hrefs_are_dropped_then_error(
        self, tmp_path: Path
    ) -> None:
        table = _editable_table()
        cast(dict[str, Any], table.at[0, 'assets'])['B08'].pop('href')
        table.to_parquet(tmp_path / 'items.parquet')
        with pytest.warns(UserWarning, match='Dropping 1 STAC item'):
            assert len(STACDataset(tmp_path / 'items.parquet', ('B04', 'B08'))) == 3
        for idx in table.index:
            for asset in cast(dict[str, Any], table.at[idx, 'assets']).values():
                asset.pop('href', None)
        table.to_parquet(tmp_path / 'items.parquet')
        with pytest.raises(ValueError, match='All STAC items were dropped'):
            STACDataset(tmp_path / 'items.parquet', ('B04', 'B08'))

    def test_unknown_asset_key_raises(self) -> None:
        with pytest.raises(ValueError, match='B11'):
            STACDataset(ITEMS, ('B11',))

    def test_transforms_run_and_signing_uses_the_handle_cache(self) -> None:
        signed: list[str] = []
        dataset = STACDataset(
            ITEMS,
            ('B04',),
            sign_href=lambda href: signed.append(href) or href,
            transforms=lambda sample: {**sample, 'transformed': True},
            crs=CRS.from_epsg(32632),
            res=10,
        )
        query = _sample_query(dataset)
        sample = dataset[query]
        dataset[query]
        assert sample['transformed'] is True
        assert signed == [cast(str, dataset.index.iloc[0]['B04'])]

    def test_getitem_out_of_bounds_raises(self) -> None:
        dataset = STACDataset(ITEMS, ('B04',))
        _, _, time = dataset.bounds
        xres, yres = dataset.res
        with pytest.raises(IndexError, match='not found in dataset'):
            dataset[
                slice(0, xres, xres),
                slice(0, yres, yres),
                slice(time.start, time.stop, 1),
            ]

    def test_missing_geometry_crs_still_uses_top_level_proj_crs(
        self, tmp_path: Path
    ) -> None:
        table = _editable_table()
        table = GeoDataFrame(
            table.drop(columns=table.geometry.name), geometry=table.geometry.to_numpy()
        )
        table.to_parquet(tmp_path / 'items.parquet')
        with pytest.warns(UserWarning, match='defaulting to EPSG:4326'):
            dataset = STACDataset(tmp_path / 'items.parquet', ('B04',), res=0.0001)
        assert dataset.crs.to_epsg() == 32632

    @pytest.mark.parametrize(
        ('href', 'index_path', 'expected'),
        [
            (
                'rasters/a.tif',
                's3://bucket/p/items.parquet',
                's3://bucket/p/rasters/a.tif',
            ),
            ('rasters/a.tif', 'C:\\d\\items.parquet', 'C:\\d\\rasters\\a.tif'),
            ('rasters/a.tif', 'C:\\d\\', 'C:\\d\\rasters\\a.tif'),
        ],
    )
    def test_resolve_relative_hrefs(
        self, href: str, index_path: str, expected: str
    ) -> None:
        assert STACDataset._resolve_href(href, index_path) == expected
