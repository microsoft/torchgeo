# Copyright (c) Microsoft Corporation. All rights reserved.
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
        assert isinstance(x['label'], torch.Tensor)

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

        # Test with both suptitle and show_titles=False
        fig = dataset.plot(sample, show_titles=False, suptitle='Custom title')
        assert isinstance(fig, Figure)
        plt.close()
