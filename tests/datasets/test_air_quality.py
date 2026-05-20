# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import pytest
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import Tensor

from torchgeo.datasets import AirQuality, DatasetNotFoundError


class TestAirQuality:
    @pytest.fixture()
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> AirQuality:
        url = os.path.join('tests', 'data', 'air_quality', 'data.csv')
        monkeypatch.setattr(AirQuality, 'url', url)
        return AirQuality(tmp_path, download=True)

    def test_getitem(self, dataset: AirQuality) -> None:
        item = dataset[0]
        x = item['x_input']
        y = item['y_target']
        assert isinstance(x, Tensor)
        assert x.shape[1] == 12
        assert x.shape[0] == dataset.num_input_steps
        assert isinstance(y, Tensor)
        assert y.shape[1] == 12
        assert y.shape[0] == dataset.num_target_steps

    def test_len(self, dataset: AirQuality) -> None:
        assert len(dataset) == 46

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            AirQuality(tmp_path)
