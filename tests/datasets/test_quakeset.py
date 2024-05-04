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

import torchgeo.datasets.utils
from torchgeo.datasets import DatasetNotFoundError, QuakeSet

from .utils import importandskip


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestQuakeSet:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> QuakeSet:
        pytest.importorskip('h5py', minversion='3.6')
        monkeypatch.setattr(torchgeo.datasets.quakeset, 'download_url', download_url)
        url = os.path.join('tests', 'data', 'quakeset', 'earthquakes.h5')
        md5 = '127d0d6a1f82d517129535f50053a4c9'
        monkeypatch.setattr(QuakeSet, 'md5', md5)
        monkeypatch.setattr(QuakeSet, 'url', url)
        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return QuakeSet(
            root, split, transforms=transforms, download=True, checksum=True
        )

    def test_missing_module(self) -> None:
        importandskip('h5py')
        root = os.path.join('tests', 'data', 'quakeset')
        with pytest.raises(
            ImportError,
            match='h5py is not installed and is required to use this dataset',
        ):
            QuakeSet(root)

    def test_getitem(self, dataset: QuakeSet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].shape[0] == 4

    def test_len(self, dataset: QuakeSet) -> None:
        assert len(dataset) == 8

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pytest.importorskip('h5py', minversion='3.6')
        root = os.path.join('tests', 'data', 'quakeset')
        QuakeSet(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        pytest.importorskip('h5py', minversion='3.6')
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            QuakeSet(str(tmp_path))

    def test_plot(self, dataset: QuakeSet) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        x['magnitude'] = torch.tensor(0.0)
        dataset.plot(x)
        plt.close()
