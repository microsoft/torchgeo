# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, EverWatch


class TestEverWatch:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> EverWatch:
        data_dir = os.path.join('tests', 'data', 'everwatch')
        url = os.path.join(data_dir, 'everwatch-benchmark.zip')
        md5 = '6d797a56dc7edea89109b38c47c55e53'
        monkeypatch.setattr(EverWatch, 'url', url)
        monkeypatch.setattr(EverWatch, 'md5', md5)
        root = tmp_path
        transforms = nn.Identity()
        return EverWatch(root=root, transforms=transforms, download=True, checksum=True)

    def test_already_downloaded(self, dataset: EverWatch) -> None:
        EverWatch(root=dataset.root, download=True)

    def test_getitem(self, dataset: EverWatch) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert isinstance(x['bbox_xyxy'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert x['image'].ndim == 3

    def test_len(self, dataset: EverWatch) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 2

    def test_not_extracted(self, tmp_path: Path) -> None:
        url = os.path.join('tests', 'data', 'everwatch', 'everwatch-benchmark.zip')
        shutil.copy(url, tmp_path)
        EverWatch(root=tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'everwatch-benchmark.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            EverWatch(root=tmp_path, checksum=True)

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EverWatch(tmp_path)

    def test_plot(self, dataset: EverWatch) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
