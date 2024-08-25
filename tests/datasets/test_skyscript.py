# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, SkyScript


class TestSkyScript:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> SkyScript:
        url = os.path.join('tests', 'data', 'skyscript', '{}')
        monkeypatch.setattr(SkyScript, 'url', url)
        transforms = nn.Identity()
        return SkyScript(tmp_path, transforms=transforms, download=True)

    def test_getitem(self, dataset: SkyScript) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['text'], str)

    def test_len(self, dataset: SkyScript) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: SkyScript) -> None:
        SkyScript(dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SkyScript(tmp_path)

    def test_plot(self, dataset: SkyScript) -> None:
        x = dataset[0]
        x['prediction'] = x['text']
        dataset.plot(x, suptitle='Test')
        plt.close()
