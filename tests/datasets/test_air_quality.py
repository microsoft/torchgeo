# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

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
        x, y = dataset[0]
        assert isinstance(x, Tensor)
        assert x.shape[1] == 15
        assert x.shape[0] == dataset.num_past_steps
        assert isinstance(y, Tensor)
        assert y.shape[1] == 15
        assert y.shape[0] == dataset.num_future_steps

    def test_len(self, dataset: AirQuality) -> None:
        assert len(dataset) == 46

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            AirQuality(tmp_path)
