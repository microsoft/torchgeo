# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest
from matplotlib.figure import Figure
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
        x = item['input']
        y = item['target']
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

    def test_already_downloaded(
        self, dataset: AirQuality, monkeypatch: MonkeyPatch
    ) -> None:
        # Copy the test CSV into dataset.root so os.path.exists hits True
        src = os.path.join('tests', 'data', 'air_quality', 'data.csv')
        dst = os.path.join(dataset.root, AirQuality.data_file_name)
        shutil.copy(src, dst)
        AirQuality(dataset.root)

    def test_plot(self, dataset: AirQuality) -> None:
        sample = dataset[0]

        fig = dataset.plot(sample)
        assert isinstance(fig, Figure)
        plt.close()

        single_feature_dataset = AirQuality.__new__(AirQuality)
        single_feature_dataset.num_input_steps = dataset.num_input_steps
        single_feature_dataset.num_target_steps = dataset.num_target_steps
        single_feature_dataset.feature_names = [dataset.feature_names[0]]

        single_sample = {
            'input': sample['input'][:, :1],
            'target': sample['target'][:, :1],
        }
        fig = single_feature_dataset.plot(single_sample)
        assert isinstance(fig, Figure)
        plt.close()
