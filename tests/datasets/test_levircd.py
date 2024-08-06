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
from torchgeo.datasets import LEVIRCD, DatasetNotFoundError, LEVIRCDPlus


def download_url(url: str, root: str | Path, *args: str) -> None:
    shutil.copy(url, root)


class TestLEVIRCD:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> LEVIRCD:
        directory = os.path.join('tests', 'data', 'levircd', 'levircd')
        splits = {
            'train': {
                'url': os.path.join(directory, 'train.zip'),
                'filename': 'train.zip',
                'md5': '7c2e24b3072095519f1be7eb01fae4ff',
            },
            'val': {
                'url': os.path.join(directory, 'val.zip'),
                'filename': 'val.zip',
                'md5': '5c320223ba88b6fc8ff9d1feebc3b84e',
            },
            'test': {
                'url': os.path.join(directory, 'test.zip'),
                'filename': 'test.zip',
                'md5': '021db72d4486726d6a0702563a617b32',
            },
        }
        monkeypatch.setattr(torchgeo.datasets.utils, 'download_url', download_url)
        monkeypatch.setattr(LEVIRCD, 'splits', splits)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return LEVIRCD(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: LEVIRCD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image1'], torch.Tensor)
        assert isinstance(x['image2'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image1'].shape[0] == 3
        assert x['image2'].shape[0] == 3

    def test_len(self, dataset: LEVIRCD) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: LEVIRCD) -> None:
        LEVIRCD(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LEVIRCD(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            LEVIRCD(tmp_path)

    def test_plot(self, dataset: LEVIRCD) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Prediction')
        plt.close()


class TestLEVIRCDPlus:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> LEVIRCDPlus:
        monkeypatch.setattr(torchgeo.datasets.utils, 'download_url', download_url)
        md5 = '0ccca34310bfe7096dadfbf05b0d180f'
        monkeypatch.setattr(LEVIRCDPlus, 'md5', md5)
        url = os.path.join('tests', 'data', 'levircd', 'levircdplus', 'LEVIR-CD+.zip')
        monkeypatch.setattr(LEVIRCDPlus, 'url', url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return LEVIRCDPlus(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: LEVIRCDPlus) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image1'], torch.Tensor)
        assert isinstance(x['image2'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image1'].shape[0] == 3
        assert x['image2'].shape[0] == 3

    def test_len(self, dataset: LEVIRCDPlus) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: LEVIRCDPlus) -> None:
        LEVIRCDPlus(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            LEVIRCDPlus(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            LEVIRCDPlus(tmp_path)

    def test_plot(self, dataset: LEVIRCDPlus) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Prediction')
        plt.close()
