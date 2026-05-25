# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch

from torchgeo.datamodules import (
    AuroraWeatherBench2Sequence,
    WeatherBench2AuroraDataModule,
    aurora_batch_from_xarray,
    aurora_collate_fn,
    aurora_predictions_to_xarray,
)
from torchgeo.datasets import DatasetNotFoundError, WeatherBench2

pytest.importorskip('aurora')
pytest.importorskip('xarray', minversion='0.17')
pytest.importorskip('zarr', minversion='2.16')


def _make_dataset() -> Any:
    fixture = Path('tests/data/weatherbench2_era5_zarr/data.py')
    spec = importlib.util.spec_from_file_location('wb2_data', fixture)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['wb2_data'] = module
    spec.loader.exec_module(module)
    return module.make_dataset()


@pytest.fixture
def dataset(monkeypatch: pytest.MonkeyPatch) -> WeatherBench2:
    data = _make_dataset()
    monkeypatch.setattr('xarray.open_zarr', lambda *a, **kw: data)
    return WeatherBench2('mock://era5.zarr')


class TestWeatherBench2:
    def test_data(self, dataset: WeatherBench2) -> None:
        assert '2m_temperature' in dataset.data.data_vars
        assert 'temperature' in dataset.data.data_vars

    def test_plot(self, dataset: WeatherBench2) -> None:
        time = dataset.data.time.values[0]
        _, ax = plt.subplots()
        dataset.plot('2m_temperature', time=time, suptitle='surf', ax=ax)
        dataset.plot('temperature', time=time, level=500.0)
        dataset.plot('2m_temperature', time=time, region=(0, -45, 90, 45))
        plt.close('all')

    def test_gs_default_storage_options(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, dict[str, str] | None] = {}
        data = _make_dataset()

        def fake_open_zarr(path: object, storage_options: object = None) -> object:
            captured['storage_options'] = storage_options  # type: ignore[assignment]
            return data

        monkeypatch.setattr('xarray.open_zarr', fake_open_zarr)
        ds = WeatherBench2('gs://example/store.zarr')
        assert ds.storage_options == {'token': 'anon'}
        assert captured['storage_options'] == {'token': 'anon'}

    def test_no_data(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fail(*a: object, **kw: object) -> object:
            raise OSError('no such file')

        monkeypatch.setattr('xarray.open_zarr', fail)
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            WeatherBench2('missing.zarr')


class TestAuroraBatchFromXarray:
    def test_global(self, dataset: WeatherBench2) -> None:
        from aurora import Batch

        times = dataset.data.time.values[:2]
        batch = aurora_batch_from_xarray(dataset.data, times)
        assert isinstance(batch, Batch)
        assert next(iter(batch.surf_vars.values())).ndim == 4
        assert next(iter(batch.atmos_vars.values())).ndim == 5
        assert next(iter(batch.static_vars.values())).ndim == 2

    def test_region(self, dataset: WeatherBench2) -> None:
        times = dataset.data.time.values[:2]
        batch = aurora_batch_from_xarray(
            dataset.data, times, region=(0.0, -45.0, 100.0, 45.0)
        )
        assert (
            next(iter(batch.surf_vars.values())).shape[-1] < dataset.data.longitude.size
        )


class TestAuroraWeatherBench2Sequence:
    def test_window(self, dataset: WeatherBench2) -> None:
        from aurora import Batch

        seq = AuroraWeatherBench2Sequence(
            dataset, start_time='2023-01-01 00:00', end_time='2023-01-01 18:00'
        )
        item = seq[0]
        assert isinstance(item['context'], Batch)
        assert isinstance(item['target'], Batch)
        assert next(iter(item['context'].surf_vars.values())).shape[1] == 2
        assert next(iter(item['target'].surf_vars.values())).shape[1] == 1
        assert item['target'].static_vars == {}

    def test_invalid_steps(self, dataset: WeatherBench2) -> None:
        with pytest.raises(ValueError, match='must be >= 1'):
            AuroraWeatherBench2Sequence(
                dataset,
                start_time='2023-01-01 00:00',
                end_time='2023-01-01 18:00',
                context_steps=0,
            )

    def test_no_valid_window(self, dataset: WeatherBench2) -> None:
        with pytest.raises(ValueError, match='No window'):
            AuroraWeatherBench2Sequence(
                dataset, start_time='2050-01-01 00:00', end_time='2050-01-02 00:00'
            )


class TestAuroraCollateFn:
    def test_collate(self, dataset: WeatherBench2) -> None:
        from aurora import Batch

        seq = AuroraWeatherBench2Sequence(
            dataset, start_time='2023-01-01 00:00', end_time='2023-01-01 18:00'
        )
        out = aurora_collate_fn([seq[0], seq[0]])
        assert isinstance(out['context'], Batch)
        assert next(iter(out['context'].surf_vars.values())).shape[0] == 2


class TestAuroraPredictionsToXarray:
    @pytest.fixture
    def batch(self, dataset: WeatherBench2) -> object:
        seq = AuroraWeatherBench2Sequence(
            dataset, start_time='2023-01-01 00:00', end_time='2023-01-01 18:00'
        )
        return seq[0]['context']

    def test_basic(self, batch: object) -> None:
        ds = aurora_predictions_to_xarray(
            [batch, batch], pd.Timestamp('2023-01-01 06:00')
        )
        assert '2m_temperature' in ds.data_vars
        assert 'temperature' in ds.data_vars
        assert ds.time.size == 2

    def test_skip_atmos(self, batch: object) -> None:
        ds = aurora_predictions_to_xarray(
            [batch], pd.Timestamp('2023-01-01'), atmos_vars={}
        )
        assert 'temperature' not in ds.data_vars

    def test_skip_surf(self, batch: object) -> None:
        ds = aurora_predictions_to_xarray(
            [batch], pd.Timestamp('2023-01-01'), surf_vars={}
        )
        assert '2m_temperature' not in ds.data_vars

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match='preds must not be empty'):
            aurora_predictions_to_xarray([], pd.Timestamp('2023-01-01'))


class TestWeatherBench2AuroraDataModule:
    @pytest.fixture
    def datamodule(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> WeatherBench2AuroraDataModule:
        data = _make_dataset()
        monkeypatch.setattr('xarray.open_zarr', lambda *a, **kw: data)
        return WeatherBench2AuroraDataModule(
            paths='mock://era5.zarr',
            start_time='2023-01-01 00:00',
            end_time='2023-01-01 18:00',
        )

    @pytest.mark.parametrize(
        'stage,loader',
        [
            ('fit', 'train_dataloader'),
            ('validate', 'val_dataloader'),
            ('test', 'test_dataloader'),
            ('predict', 'predict_dataloader'),
        ],
    )
    def test_stages(
        self, datamodule: WeatherBench2AuroraDataModule, stage: str, loader: str
    ) -> None:
        datamodule.setup(stage)
        out = next(iter(getattr(datamodule, loader)()))
        assert 'context' in out and 'target' in out
        for tensor in out['target'].surf_vars.values():
            assert isinstance(tensor, torch.Tensor)

    def test_on_after_batch_transfer(
        self, datamodule: WeatherBench2AuroraDataModule
    ) -> None:
        datamodule.setup('predict')
        batch = next(iter(datamodule.predict_dataloader()))
        assert datamodule.on_after_batch_transfer(batch, 0) is batch
