# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import SSL4EOL, SSL4EOS12, DatasetNotFoundError


class TestSSL4EOL:
    @pytest.fixture(params=zip(SSL4EOL.metadata.keys(), [1, 1, 2, 2, 4]))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SSL4EOL:
        url = os.path.join('tests', 'data', 'ssl4eo', 'l', 'ssl4eo_l_{0}.tar.gz{1}')
        monkeypatch.setattr(SSL4EOL, 'url', url)

        checksums = {
            'tm_toa': {'aa': '', 'ab': '', 'ac': ''},
            'etm_toa': {'aa': '', 'ab': '', 'ac': ''},
            'etm_sr': {'aa': '', 'ab': '', 'ac': ''},
            'oli_tirs_toa': {'aa': '', 'ab': '', 'ac': ''},
            'oli_sr': {'aa': '', 'ab': '', 'ac': ''},
        }
        monkeypatch.setattr(SSL4EOL, 'checksums', checksums)

        root = tmp_path
        split, seasons = request.param
        transforms = nn.Identity()
        return SSL4EOL(root, split, seasons, transforms, download=True)

    def test_getitem(self, dataset: SSL4EOL) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].size(0) == dataset.seasons * len(
            dataset.metadata[dataset.split]['all_bands']
        )

    def test_len(self, dataset: SSL4EOL) -> None:
        assert len(dataset) == 2

    def test_add(self, dataset: SSL4EOL) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2 * 2

    def test_already_extracted(self, dataset: SSL4EOL) -> None:
        SSL4EOL(dataset.root, dataset.split, dataset.seasons)

    def test_already_downloaded(self, dataset: SSL4EOL, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'ssl4eo', 'l', '*.tar.gz*')
        root = tmp_path
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        SSL4EOL(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SSL4EOL(tmp_path)

    def test_plot(self, dataset: SSL4EOL) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle='Test')
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()


class TestSSL4EOS12:
    @pytest.fixture(params=zip(SSL4EOS12.metadata.keys(), [1, 2, 4]))
    def dataset(self, request: SubRequest) -> SSL4EOS12:
        root = os.path.join('tests', 'data', 'ssl4eo', 's12')
        split, seasons = request.param
        transforms = nn.Identity()
        return SSL4EOS12(root, split, seasons, transforms)

    def test_getitem(self, dataset: SSL4EOS12) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].size(0) == dataset.seasons * len(dataset.bands)

    def test_len(self, dataset: SSL4EOS12) -> None:
        assert len(dataset) == 251079

    def test_add(self, dataset: SSL4EOS12) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2 * 251079

    def test_download(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        url = os.path.join('tests', 'data', 'ssl4eo', 's12', '{0}.tar.gz.part{1}')
        checksums = {'s2c': {'aa': '', 'ab': ''}}
        monkeypatch.setattr(SSL4EOS12, 'url', url)
        monkeypatch.setattr(SSL4EOS12, 'checksums', checksums)
        SSL4EOS12(tmp_path, download=True)

    def test_extract(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        root = os.path.join('tests', 'data', 'ssl4eo', 's12')
        checksums = {'s2c': {'aa': '', 'ab': ''}}
        monkeypatch.setattr(SSL4EOS12, 'checksums', checksums)
        for filename in ['s2_l1c.tar.gz.partaa', 's2_l1c.tar.gz.partab']:
            shutil.copyfile(os.path.join(root, filename), tmp_path / filename)
        SSL4EOS12(tmp_path)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SSL4EOS12(tmp_path)

    def test_plot(self, dataset: SSL4EOS12) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle='Test')
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()
