# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import Tensor

from torchgeo.datasets import AirQuality, DatasetNotFoundError


class TestAirQuality:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> AirQuality:
        url = os.path.join('tests', 'data', 'air_quality', 'data.csv')
        monkeypatch.setattr(AirQuality, 'url', url)
        return AirQuality(tmp_path, download=True)

    def test_getitem(self, dataset: AirQuality) -> None:
        item = dataset[0]
        x = item['input']
        y = item['target']
        assert isinstance(x, Tensor)
        assert x.shape[1] == 17
        assert x.shape[0] == dataset.input_steps
        assert isinstance(y, Tensor)
        assert y.shape[1] == 17
        assert y.shape[0] == dataset.target_steps

    def test_len(self, dataset: AirQuality) -> None:
        assert len(dataset) == 7

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            AirQuality(tmp_path)

    def test_already_downloaded(self) -> None:
        root = os.path.join('tests', 'data', 'air_quality')
        AirQuality(root)

    def test_plot(self, dataset: AirQuality) -> None:
        sample = dataset[0]
        dataset.plot(sample)
        plt.close()
