# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import importlib
import sys
from pathlib import Path

import pytest
import torch

from torchgeo.datasets import DatasetNotFoundError, WeatherBench2

# zarr v3 uses asyncio internally, which creates local Unix sockets.
# Re-enable sockets for this module so --disable-socket doesn't block it.
pytestmark = pytest.mark.enable_socket


def _make_store(store: Path) -> Path:
    """Build a tiny WeatherBench2-like Zarr fixture under *store*."""
    pytest.importorskip('xarray', minversion='0.17')
    pytest.importorskip('zarr', minversion='2.16')

    fixture = Path('tests/data/weatherbench2_era5_zarr/data.py')
    spec = importlib.util.spec_from_file_location('wb2_data', fixture)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['wb2_data'] = module
    spec.loader.exec_module(module)
    module.main(str(store))
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
        # Passing the parent directory should also discover the .zarr store.
        _make_store(tmp_path / 'era5.zarr')
        ds = WeatherBench2(tmp_path, data_vars=('2m_temperature',))
        sample = ds[ds.bounds]
        assert isinstance(sample['image'], torch.Tensor)

    def test_storage_options_local_zarr(self, store: Path) -> None:
        # ``storage_options`` is forwarded to ``xarray.open_zarr``; local paths
        # should ignore an empty mapping.
        ds = WeatherBench2(
            store, data_vars=('2m_temperature',), storage_options={}
        )
        sample = ds[ds.bounds]
        assert isinstance(sample['image'], torch.Tensor)
