# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, WesternUSALiveFuelMoisture
from torchgeo.datasets.utils import Executable


class TestWesternUSALiveFuelMoisture:
    @pytest.fixture
    def dataset(
        self, azcopy: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> WesternUSALiveFuelMoisture:
        url = os.path.join('tests', 'data', 'western_usa_live_fuel_moisture')
        monkeypatch.setattr(WesternUSALiveFuelMoisture, 'url', url)
        transforms = nn.Identity()
        return WesternUSALiveFuelMoisture(
            tmp_path, transforms=transforms, download=True
        )

    def test_getitem(self, dataset: WesternUSALiveFuelMoisture) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['input'], torch.Tensor)
        assert x['input'].shape == (4, 34)
        assert x['input'].dtype == torch.float32
        assert isinstance(x['lon'], torch.Tensor)
        assert isinstance(x['lat'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        # time steps run t, t-1, t-2, t-3 down the rows
        vv = dataset.input_features.index('vv')
        assert x['input'][0, vv] == pytest.approx(-12.80108143, rel=1e-5)
        assert x['input'][3, vv] == pytest.approx(-12.35794964, rel=1e-5)
        assert x['lon'] == pytest.approx(-115.8855556, rel=1e-5)
        assert x['lat'] == pytest.approx(42.44111111, rel=1e-5)
        assert x['label'] == pytest.approx(132.6666667, rel=1e-5)

    def test_input_features_subset(
        self, azcopy: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> None:
        url = os.path.join('tests', 'data', 'western_usa_live_fuel_moisture')
        monkeypatch.setattr(WesternUSALiveFuelMoisture, 'url', url)
        dataset = WesternUSALiveFuelMoisture(
            tmp_path, input_features=['vv', 'ndvi'], download=True
        )
        assert dataset[0]['input'].shape == (4, 2)

    def test_invalid_input_features(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError):
            WesternUSALiveFuelMoisture(tmp_path, input_features=['not_a_variable'])

    def test_len(self, dataset: WesternUSALiveFuelMoisture) -> None:
        assert len(dataset) == 3

    def test_already_downloaded(self, dataset: WesternUSALiveFuelMoisture) -> None:
        WesternUSALiveFuelMoisture(dataset.root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            WesternUSALiveFuelMoisture(tmp_path)

    def test_plot(self, dataset: WesternUSALiveFuelMoisture) -> None:
        sample = dataset[0]

        # Test with a single variable - likely one of the missing lines
        fig = dataset.plot(sample, variables_to_plot=['vv'])
        assert isinstance(fig, Figure)
        plt.close()

        # Test with both suptitle and show_titles=False (default variables)
        fig = dataset.plot(sample, show_titles=False, suptitle='Custom title')
        assert isinstance(fig, Figure)
        plt.close()

        # Unknown variables are filtered out, leaving only the valid ones
        fig = dataset.plot(sample, variables_to_plot=['vv', 'not_a_variable'])
        assert isinstance(fig, Figure)
        plt.close()

    def test_plot_no_valid_variables(self, dataset: WesternUSALiveFuelMoisture) -> None:
        with pytest.raises(ValueError, match='input_features'):
            dataset.plot(dataset[0], variables_to_plot=['not_a_variable'])
