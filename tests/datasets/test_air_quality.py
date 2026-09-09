# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest
from pandas.testing import assert_frame_equal
from pytest import MonkeyPatch
from torch import Tensor

from torchgeo.datasets import AirQuality, DatasetNotFoundError

GT = ['CO(GT)', 'NMHC(GT)', 'C6H6(GT)', 'NOx(GT)', 'NO2(GT)']


class TestAirQuality:
    @pytest.fixture(params=[(None, None), (GT, None), (None, GT), (GT, GT)])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> AirQuality:
        url = os.path.join('tests', 'data', 'air_quality', 'data.csv')
        monkeypatch.setattr(AirQuality, 'url', url)
        input_features, target_features = request.param
        return AirQuality(
            tmp_path,
            input_features=input_features,
            target_features=target_features,
            download=True,
        )

    def test_getitem(self, dataset: AirQuality) -> None:
        item = dataset[0]
        x = item['input']
        y = item['target']
        assert isinstance(x, Tensor)
        assert x.shape[1] in {5, 17}
        assert x.shape[0] == dataset.input_steps
        assert isinstance(y, Tensor)
        assert y.shape[1] in {5, 17}
        assert y.shape[0] == dataset.target_steps

    def test_len(self, dataset: AirQuality) -> None:
        assert len(dataset) == 17

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            AirQuality(tmp_path)

    def test_already_downloaded(self) -> None:
        root = os.path.join('tests', 'data', 'air_quality')
        AirQuality(root)

    def test_download_saves_data_for_offline_reuse(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        source = Path('tests/data/air_quality/data.csv').resolve()
        monkeypatch.setattr(AirQuality, 'url', str(source))
        root = tmp_path / 'air_quality'

        downloaded = AirQuality(root, download=True)

        cached_file = root / AirQuality.data_file_name
        assert cached_file.is_file()
        assert cached_file.read_bytes() == source.read_bytes()

        monkeypatch.setattr(AirQuality, 'url', str(tmp_path / 'unavailable.csv'))
        offline = AirQuality(root, download=False)
        assert_frame_equal(offline.input_data, downloaded.input_data)
        assert_frame_equal(offline.target_data, downloaded.target_data)

    @pytest.mark.parametrize('features', [None, GT])
    def test_plot(self, dataset: AirQuality, features: list[str] | None) -> None:
        sample = dataset[0]
        sample['prediction'] = sample['target']
        dataset.plot(sample, suptitle='Test', features=features)
        plt.close()
