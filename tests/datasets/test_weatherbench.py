# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import itertools
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest
from matplotlib import pyplot as plt

from torchgeo.datasets import WeatherBench2

pytest.importorskip('xarray', minversion='0.17')
pytest.importorskip('zarr')


@pytest.mark.enable_socket  # required for asyncio
class TestWeatherBench2:
    @pytest.fixture(
        scope='class',
        params=itertools.product(
            ([], ['land_sea_mask'], ['land_sea_mask', 'soil_type']),  # 2D vars
            ([], ['2m_temperature'], ['2m_temperature', '10m_wind_speed']),  # 3D vars
            ([], ['temperature'], ['temperature', 'wind_speed']),  # 4D vars
        ),
    )
    @classmethod
    def dataset(cls, request: SubRequest) -> WeatherBench2:
        root = Path('tests') / 'data' / 'weatherbench'
        store = root / '1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr'
        data_vars = [v for vs in request.param for v in vs]
        return WeatherBench2(store, data_vars=data_vars)

    def test_getitem(self, dataset: WeatherBench2) -> None:
        dataset[dataset.bounds]

    def test_len(self, dataset: WeatherBench2) -> None:
        assert len(dataset) == 1

    def test_plot(self, dataset: WeatherBench2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
