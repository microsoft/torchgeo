# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pandas as pd
import pytest
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import AirQuality, DatasetNotFoundError


class TestAirQuality:
    @pytest.fixture()
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> AirQuality:
        url = 'tests/data/air_quality/data.csv'
        monkeypatch.setattr(AirQuality, 'url', url)
        return AirQuality(tmp_path, download=True)

    def test_getitem(self, dataset: AirQuality) -> None:
        x, y = dataset[0]
        assert isinstance(x, pd.DataFrame)
        assert len(x.columns) == 15
        assert len(x) == dataset.past_steps
        assert isinstance(y, pd.DataFrame)
        assert len(y.columns) == 15
        assert len(y) == dataset.future_steps

    def test_len(self, dataset: AirQuality) -> None:
        assert len(dataset) == 46

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            AirQuality(tmp_path)
