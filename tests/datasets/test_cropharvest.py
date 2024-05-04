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

import torchgeo.datasets.utils
from torchgeo.datasets import CropHarvest, DatasetNotFoundError

from .utils import importandskip


def download_url(url: str, root: str, filename: str, md5: str) -> None:
    shutil.copy(url, os.path.join(root, filename))


class TestCropHarvest:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> CropHarvest:
        monkeypatch.setattr(torchgeo.datasets.cropharvest, 'download_url', download_url)
        monkeypatch.setitem(
            CropHarvest.file_dict['features'], 'md5', 'ef6f4f00c0b3b50ed8380b0044928572'
        )
        monkeypatch.setitem(
            CropHarvest.file_dict['labels'], 'md5', '1d93b6bfcec7b6797b75acbd9d284b92'
        )
        monkeypatch.setitem(
            CropHarvest.file_dict['features'],
            'url',
            os.path.join('tests', 'data', 'cropharvest', 'features.tar.gz'),
        )
        monkeypatch.setitem(
            CropHarvest.file_dict['labels'],
            'url',
            os.path.join('tests', 'data', 'cropharvest', 'labels.geojson'),
        )

        root = str(tmp_path)
        transforms = nn.Identity()

        dataset = CropHarvest(root, transforms, download=True, checksum=True)
        return dataset

    def test_getitem(self, dataset: CropHarvest) -> None:
        pytest.importorskip('h5py', minversion='3.6')
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['array'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['array'].shape == (12, 18)
        y = dataset[2]
        assert y['label'] == 1

    def test_len(self, dataset: CropHarvest) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(self, dataset: CropHarvest) -> None:
        CropHarvest(dataset.root)

    def test_downloaded_zipped(self, dataset: CropHarvest, tmp_path: Path) -> None:
        feature_path = os.path.join(tmp_path, 'features')
        shutil.rmtree(feature_path)
        CropHarvest(root=str(tmp_path))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CropHarvest(str(tmp_path))

    def test_plot(self, dataset: CropHarvest) -> None:
        pytest.importorskip('h5py', minversion='3.6')
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_missing_module(self) -> None:
        importandskip('h5py')
        root = os.path.join('tests', 'data', 'cropharvest')
        match = 'h5py is not installed and is required to use this dataset'
        with pytest.raises(ImportError, match=match):
            CropHarvest(root)[0]
