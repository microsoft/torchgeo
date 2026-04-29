# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import importlib
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from torchgeo.datamodules import (
    AuroraWeatherBench2Sequence,
    WeatherBench2AuroraDataModule,
    aurora_collate_fn,
)
from torchgeo.datasets import WeatherBench2

pytestmark = pytest.mark.enable_socket
pytest.importorskip('aurora')


def _make_store(store: Path) -> Path:
    """Build a tiny WeatherBench2-like Zarr fixture under *store*."""
    pytest.importorskip('xarray', minversion='0.17')
    pytest.importorskip('zarr', minversion='2.16')

    fixture = Path('tests/data/weatherbench2_era5_zarr/data.py')
    spec = importlib.util.spec_from_file_location('wb2_data_dm', fixture)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['wb2_data_dm'] = module
    spec.loader.exec_module(module)
    module.main(str(store))
    return store


class TestAuroraWeatherBench2Sequence:
    @pytest.fixture
    def dataset(self, tmp_path: Path) -> WeatherBench2:
        _make_store(tmp_path / 'era5.zarr')
        return WeatherBench2(tmp_path / 'era5.zarr')

    def test_window_shapes(self, dataset: WeatherBench2) -> None:
        seq = AuroraWeatherBench2Sequence(
            dataset=dataset,
            region=(0.0, -90.0, 100.0, 0.0),
            start_time='2023-01-01 00:00',
            end_time='2023-01-01 18:00',
            timestep='6h',
            context_steps=2,
            target_steps=1,
        )
        assert len(seq) >= 1

        item = seq[0]
        # Surface: [T_ctx, H, W]
        for tensor in item['surf_vars'].values():
            assert tensor.ndim == 3
            assert tensor.shape[0] == 2
        # Atmospheric: [T_ctx, L, H, W]
        for tensor in item['atmos_vars'].values():
            assert tensor.ndim == 4
            assert tensor.shape[0] == 2
        # Static: [H, W]
        for tensor in item['static_vars'].values():
            assert tensor.ndim == 2
        # Targets: T_tgt = 1
        for tensor in item['target_surf_vars'].values():
            assert tensor.shape[0] == 1
        for tensor in item['target_atmos_vars'].values():
            assert tensor.shape[0] == 1

    def test_caps_to_store_time_range(self, dataset: WeatherBench2) -> None:
        # ``end_time`` runs well past the fixture's last timestamp.
        seq = AuroraWeatherBench2Sequence(
            dataset=dataset,
            region=(0.0, -90.0, 100.0, 0.0),
            start_time='2023-01-01 00:00',
            end_time='2099-01-01 00:00',
            timestep='6h',
            context_steps=2,
            target_steps=1,
        )
        # tmax in fixture is 2023-01-01 18:00, window length is 12h, so
        # the last allowed start is 2023-01-01 06:00.
        assert seq.starts[-1] <= pd.Timestamp('2023-01-01 06:00')

    def test_no_valid_window(self, dataset: WeatherBench2) -> None:
        with pytest.raises(ValueError, match='No window'):
            AuroraWeatherBench2Sequence(
                dataset=dataset,
                region=(0.0, -90.0, 100.0, 0.0),
                start_time='2050-01-01 00:00',
                end_time='2050-01-02 00:00',
                timestep='6h',
            )


class TestAuroraCollateFn:
    @pytest.fixture
    def dataset(self, tmp_path: Path) -> WeatherBench2:
        _make_store(tmp_path / 'era5.zarr')
        return WeatherBench2(tmp_path / 'era5.zarr')

    def test_collate(self, dataset: WeatherBench2) -> None:
        from aurora import Batch

        seq = AuroraWeatherBench2Sequence(
            dataset=dataset,
            region=(0.0, -90.0, 100.0, 0.0),
            start_time='2023-01-01 00:00',
            end_time='2023-01-01 18:00',
            timestep='6h',
        )
        out = aurora_collate_fn([seq[0]])
        assert isinstance(out['batch'], Batch)
        # surf_vars shape: [B=1, T=2, H, W]
        assert next(iter(out['batch'].surf_vars.values())).ndim == 4
        # atmos_vars shape: [B=1, T=2, L, H, W]
        assert next(iter(out['batch'].atmos_vars.values())).ndim == 5
        # static_vars shape: [H, W]
        assert next(iter(out['batch'].static_vars.values())).ndim == 2
        # Targets keep [B=1, T=1, ...]
        assert next(iter(out['target_surf_vars'].values())).shape[1] == 1
        assert next(iter(out['target_atmos_vars'].values())).shape[1] == 1


class TestWeatherBench2AuroraDataModule:
    @pytest.fixture
    def datamodule(self, tmp_path: Path) -> WeatherBench2AuroraDataModule:
        _make_store(tmp_path / 'era5.zarr')
        return WeatherBench2AuroraDataModule(
            paths=str(tmp_path / 'era5.zarr'),
            region=(0.0, -90.0, 100.0, 0.0),
            start_time='2023-01-01 00:00',
            end_time='2023-01-01 18:00',
            timestep='6h',
            context_steps=2,
            target_steps=1,
            batch_size=1,
            num_workers=0,
        )

    def test_train_dataloader(self, datamodule: WeatherBench2AuroraDataModule) -> None:
        from aurora import Batch

        datamodule.setup('fit')
        out = next(iter(datamodule.train_dataloader()))
        assert isinstance(out['batch'], Batch)
        assert 'target_surf_vars' in out
        assert 'target_atmos_vars' in out
        for tensor in out['target_surf_vars'].values():
            assert isinstance(tensor, torch.Tensor)
            assert tensor.ndim == 4

    def test_val_dataloader(self, datamodule: WeatherBench2AuroraDataModule) -> None:
        datamodule.setup('validate')
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: WeatherBench2AuroraDataModule) -> None:
        datamodule.setup('test')
        next(iter(datamodule.test_dataloader()))
