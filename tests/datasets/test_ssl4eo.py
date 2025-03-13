# Copyright (c) Microsoft Corporation. All rights reserved.
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
            'tm_toa': {
                'aa': '010b9d72b476e0e30741c17725f84e5c',
                'ab': '39171bd7bca8a56a8cb339a0f88da9d3',
                'ac': '3cfc407ce3f4f4d6e3c5fdb457bb87da',
            },
            'etm_toa': {
                'aa': '87e47278f5a30acd3b696b6daaa4713b',
                'ab': '59295e1816e08a5acd3a18ae56b6f32e',
                'ac': 'f3ff76eb6987501000228ce15684e09f',
            },
            'etm_sr': {
                'aa': 'fd61a4154eafaeb350dbb01a2551a818',
                'ab': '0c3117bc7682ba9ffdc6871e6c364b36',
                'ac': '93d3385e47de4578878ca5c4fa6a628d',
            },
            'oli_tirs_toa': {
                'aa': 'defb9e91a73b145b2dbe347649bded06',
                'ab': '97f7edaa4e288fc14ec7581dccea766f',
                'ac': '7472fad9929a0dc96ccf4dc6c804b92f',
            },
            'oli_sr': {
                'aa': '8fd3aa6b581d024299f44457956faa05',
                'ab': '7eb4d761ce1afd89cae9c6142ca17882',
                'ac': 'a3210da9fcc71e3a4efde71c30d78c59',
            },
        }
        monkeypatch.setattr(SSL4EOL, 'checksums', checksums)

        root = tmp_path
        split, seasons = request.param
        transforms = nn.Identity()
        return SSL4EOL(root, split, seasons, transforms, download=True, checksum=True)

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
        checksums = {
            's2c': {
                'aa': 'ddabb8a21c75bc10c047ca480d0f93c9',
                'ab': '5e634bb5bf4c261ce6f8c46afcccf2d5',
            }
        }
        monkeypatch.setattr(SSL4EOS12, 'url', url)
        monkeypatch.setattr(SSL4EOS12, 'checksums', checksums)
        SSL4EOS12(tmp_path, download=True)

    def test_extract(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        root = os.path.join('tests', 'data', 'ssl4eo', 's12')
        checksums = {
            's2c': {
                'aa': 'ddabb8a21c75bc10c047ca480d0f93c9',
                'ab': '5e634bb5bf4c261ce6f8c46afcccf2d5',
            }
        }
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
