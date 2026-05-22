# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch

from torchgeo.datasets import DatasetNotFoundError, WeatherBench2

# zarr v3 uses asyncio internally, which creates local Unix sockets.
# Re-enable sockets for this module so --disable-socket doesn't block it.
pytestmark = pytest.mark.enable_socket


def _load_fixture_module() -> Any:
    """Load ``tests/data/weatherbench/data.py`` as an importable module."""
    pytest.importorskip('xarray', minversion='0.17')
    pytest.importorskip('zarr', minversion='2.16')

    fixture = Path('tests/data/weatherbench/data.py')
    spec = importlib.util.spec_from_file_location('wb2_data', fixture)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['wb2_data'] = module
    spec.loader.exec_module(module)
    return module


def _make_store(store: Path, **kwargs: Any) -> Path:
    """Build a tiny WeatherBench2-like Zarr fixture under *store*."""
    module = _load_fixture_module()
    module.main(str(store), **kwargs)
    return store


class TestWeatherBench2:
    @pytest.fixture
    def store(self, tmp_path: Path) -> Path:
        return _make_store(tmp_path / 'era5.zarr')

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            WeatherBench2(tmp_path / 'missing.zarr')

    def test_getitem(self, store: Path) -> None:
        ds = WeatherBench2(store, data_vars=('2m_temperature',))
        sample = ds[ds.bounds]
        assert isinstance(sample, dict)
        assert isinstance(sample['image'], torch.Tensor)
        assert sample['image'].shape[0] == 1
        assert sample['lat'].ndim == 1
        assert sample['lon'].ndim == 1
        assert isinstance(sample['time'], tuple)

    def test_variable_ordering(self, store: Path) -> None:
        order = ('2m_temperature', '10m_u_component_of_wind')
        ds = WeatherBench2(store, data_vars=order)
        sample = ds[ds.bounds]
        assert tuple(sample['variables'].keys()) == order
        assert sample['image'].shape[0] == len(order)

    def test_atmos_levels(self, store: Path) -> None:
        ds = WeatherBench2(store, data_vars=('temperature',))
        sample = ds[ds.bounds]
        assert sample['atmos_levels'] == (50.0, 250.0, 500.0, 1000.0)

    def test_image_omitted_for_mixed_shapes(self, store: Path) -> None:
        ds = WeatherBench2(
            store, data_vars=('2m_temperature', 'temperature', 'land_sea_mask')
        )
        sample = ds[ds.bounds]
        assert 'image' not in sample
        assert set(sample['variables']) == {
            '2m_temperature',
            'temperature',
            'land_sea_mask',
        }

    def test_directory_path(self, tmp_path: Path) -> None:
        _make_store(tmp_path / 'era5.zarr')
        ds = WeatherBench2(tmp_path, data_vars=('2m_temperature',))
        sample = ds[ds.bounds]
        assert isinstance(sample['image'], torch.Tensor)

    def test_storage_options_local_zarr(self, store: Path) -> None:
        # ``storage_options`` is forwarded to ``xarray.open_zarr`` only for
        # remote paths; local paths should ignore the mapping entirely.
        ds = WeatherBench2(store, data_vars=('2m_temperature',), storage_options={})
        sample = ds[ds.bounds]
        assert isinstance(sample['image'], torch.Tensor)

    def test_scalar_resolution(self, store: Path) -> None:
        # Passing ``res`` as a scalar should be normalized to ``(res, res)``
        # by both :meth:`__init__` and :meth:`_spatial_metadata`.
        ds = WeatherBench2(store, res=0.5, data_vars=('2m_temperature',))
        assert ds.res == (0.5, 0.5)

    def test_skips_unreadable_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _make_store(tmp_path / 'good.zarr')
        bad = tmp_path / 'bad.zarr'
        bad.mkdir()

        xr = pytest.importorskip('xarray')
        real_open_zarr = xr.open_zarr

        def fake_open_zarr(path: Any, **kwargs: Any) -> Any:
            if 'bad.zarr' in str(path):
                raise OSError('cannot read store')
            return real_open_zarr(path, **kwargs)

        monkeypatch.setattr(xr, 'open_zarr', fake_open_zarr)

        ds = WeatherBench2(tmp_path, data_vars=('2m_temperature',))
        assert len(ds.index) == 1
        assert ds.index.iloc[0]['filepath'].endswith('good.zarr')

    def test_index_error_outside_time_range(self, store: Path) -> None:
        ds = WeatherBench2(store, data_vars=('2m_temperature',))
        x, y, _ = ds.bounds
        far_future = slice(pd.Timestamp('2050-01-01'), pd.Timestamp('2050-01-02'))
        with pytest.raises(IndexError, match='not found in dataset'):
            ds[x, y, far_future]

    def test_transforms_applied(self, store: Path) -> None:
        sentinel: dict[str, Any] = {'tagged': True}

        def transform(sample: dict[str, Any]) -> dict[str, Any]:
            return {**sample, **sentinel}

        ds = WeatherBench2(store, data_vars=('2m_temperature',), transforms=transform)
        sample = ds[ds.bounds]
        assert sample['tagged'] is True

    def test_unknown_variable_skipped(self, store: Path) -> None:
        ds = WeatherBench2(store, data_vars=('2m_temperature', 'does_not_exist'))
        sample = ds[ds.bounds]
        assert set(sample['variables']) == {'2m_temperature'}

    def test_ascending_latitude(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path / 'era5.zarr', descending_lat=False)
        ds = WeatherBench2(store, data_vars=('2m_temperature',))
        sample = ds[ds.bounds]
        assert sample['lat'][0].item() < sample['lat'][-1].item()

    def test_descending_longitude(self, tmp_path: Path) -> None:
        store = _make_store(tmp_path / 'era5.zarr', descending_lon=True)
        ds = WeatherBench2(store, data_vars=('2m_temperature',))
        sample = ds[ds.bounds]
        assert sample['lon'][0].item() > sample['lon'][-1].item()

    def test_multi_store_concat(self, tmp_path: Path) -> None:
        _make_store(tmp_path / 'a.zarr', start_date='2023-01-01')
        _make_store(tmp_path / 'b.zarr', start_date='2023-02-01')
        ds = WeatherBench2(tmp_path, data_vars=('2m_temperature',))
        sample = ds[ds.bounds]
        # Both stores contribute their PERIODS=4 timestamps after concat.
        assert len(sample['time']) == 8

    def test_paths_as_list(self, tmp_path: Path) -> None:
        a = _make_store(tmp_path / 'a.zarr', start_date='2023-01-01')
        b = _make_store(tmp_path / 'b.zarr', start_date='2023-02-01')
        ds = WeatherBench2([a, b], data_vars=('2m_temperature',))
        assert len(ds.index) == 2

    def test_open_xarray_default_module(self, store: Path) -> None:
        with WeatherBench2._open_xarray(str(store)) as opened:
            assert '2m_temperature' in opened.data_vars

    def test_spatial_metadata_missing_coords(self) -> None:
        xr = pytest.importorskip('xarray')
        empty = xr.Dataset()
        with pytest.raises(ValueError, match='missing longitude'):
            WeatherBench2._spatial_metadata(empty, None, None)

    def test_spatial_metadata_too_few_coords(self) -> None:
        xr = pytest.importorskip('xarray')
        sparse = xr.Dataset(coords={'longitude': [0.0], 'latitude': [0.0]})
        with pytest.raises(ValueError, match='fewer than 2'):
            WeatherBench2._spatial_metadata(sparse, None, None)

    def test_remote_uri(self, store: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Stand-in for a public WeatherBench2 store on GCS: the dataset
        # should treat ``gs://`` as opaque and forward ``storage_options``
        # to :func:`xarray.open_zarr`.
        pytest.importorskip('fsspec')
        pytest.importorskip('gcsfs')
        xr = pytest.importorskip('xarray')

        real_open_zarr = xr.open_zarr
        captured: dict[str, Any] = {}

        def fake_open_zarr(path: Any, **kwargs: Any) -> Any:
            captured.setdefault('paths', []).append(str(path))
            captured['kwargs'] = kwargs
            return real_open_zarr(str(store))

        monkeypatch.setattr(xr, 'open_zarr', fake_open_zarr)

        remote = 'gs://fake-bucket/era5.zarr'
        ds = WeatherBench2(
            remote, data_vars=('2m_temperature',), storage_options={'token': 'anon'}
        )
        assert ds.files == [remote]
        assert captured['kwargs'].get('storage_options') == {'token': 'anon'}
        sample = ds[ds.bounds]
        assert isinstance(sample['image'], torch.Tensor)
