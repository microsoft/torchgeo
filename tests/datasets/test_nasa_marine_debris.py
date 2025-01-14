# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, NASAMarineDebris
from torchgeo.datasets.utils import Executable


class TestNASAMarineDebris:
    @pytest.fixture
    def dataset(
        self, azcopy: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> NASAMarineDebris:
        url = os.path.join('tests', 'data', 'nasa_marine_debris')
        monkeypatch.setattr(NASAMarineDebris, 'url', url)
        transforms = nn.Identity()
        return NASAMarineDebris(tmp_path, transforms, download=True)

    def test_getitem(self, dataset: NASAMarineDebris) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert x['bbox_xyxy'].shape[-1] == 4

    def test_len(self, dataset: NASAMarineDebris) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(
        self, dataset: NASAMarineDebris, tmp_path: Path
    ) -> None:
        NASAMarineDebris(tmp_path, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            NASAMarineDebris(tmp_path)

    def test_plot(self, dataset: NASAMarineDebris) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction_bbox_xyxy'] = x['bbox_xyxy'].clone()
        dataset.plot(x)
        plt.close()
