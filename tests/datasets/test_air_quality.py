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
        x = dataset[0]
        assert isinstance(x, pd.Series)
        assert len(x) == 15

    def test_len(self, dataset: AirQuality) -> None:
        assert len(dataset) == 3

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            AirQuality(tmp_path)
