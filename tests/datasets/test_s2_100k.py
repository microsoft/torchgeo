# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.nn import Identity

from torchgeo.datasets import DatasetNotFoundError, S2100k


class TestS2100k:
    @pytest.fixture(params=['both', 'points'])
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> S2100k:
        root = Path('tests', 'data', 's2_100k')
        mode = request.param
        monkeypatch.setattr(S2100k, 'url', root)
        transforms = Identity()
        return S2100k(root, mode=mode, transforms=transforms)

    def test_getitem(self, dataset: S2100k) -> None:
        x = dataset[0]
        assert x['point'].shape == (2,)
        if 'image' in x:
            assert x['image'].shape == (12, 32, 32)

    def test_len(self, dataset: S2100k) -> None:
        assert len(dataset) == 1

    def test_download(self, dataset: S2100k, tmp_path: Path) -> None:
        S2100k(tmp_path, download=True)

    def test_extract(self, dataset: S2100k, tmp_path: Path) -> None:
        (tmp_path / 'images').mkdir()
        shutil.copy(dataset.root / 'index.csv', tmp_path)
        shutil.copy(dataset.root / 'satclip.tar', tmp_path)
        S2100k(tmp_path)

    def test_not_found(self, dataset: S2100k, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            S2100k(tmp_path)

        shutil.copy(dataset.root / 'index.csv', tmp_path)
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            S2100k(tmp_path)

    def test_plot(self, dataset: S2100k) -> None:
        x = dataset[0]
        if dataset.mode == 'both':
            dataset.plot(x, suptitle='Test')
            plt.close()
