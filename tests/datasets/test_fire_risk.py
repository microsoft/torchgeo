# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, FireRisk


class TestFireRisk:
    @pytest.fixture(params=['train', 'val'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> FireRisk:
        url = os.path.join('tests', 'data', 'fire_risk', 'FireRisk.zip')
        md5 = 'db22106d61b10d855234b4a74db921ac'
        monkeypatch.setattr(FireRisk, 'md5', md5)
        monkeypatch.setattr(FireRisk, 'url', url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return FireRisk(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: FireRisk) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].shape[0] == 3

    def test_len(self, dataset: FireRisk) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(self, dataset: FireRisk, tmp_path: Path) -> None:
        FireRisk(root=tmp_path, download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: FireRisk, tmp_path: Path
    ) -> None:
        shutil.rmtree(os.path.dirname(dataset.root))
        shutil.copy(dataset.url, tmp_path)
        FireRisk(root=tmp_path, download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            FireRisk(tmp_path)

    def test_plot(self, dataset: FireRisk) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()
