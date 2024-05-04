# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import builtins
import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

import torchgeo.datasets.utils
from torchgeo.datasets import CropHarvest, DatasetNotFoundError

pytest.importorskip('h5py', minversion='3.6')


def download_url(url: str, root: str, filename: str, md5: str) -> None:
    shutil.copy(url, os.path.join(root, filename))


class TestCropHarvest:
    @pytest.fixture
    def mock_missing_module(self, monkeypatch: MonkeyPatch) -> None:
        import_orig = builtins.__import__

        def mocked_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == 'h5py':
                raise ImportError()
            return import_orig(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', mocked_import)

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
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['array'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['array'].shape == (12, 18)
        y = dataset[2]
        assert y['label'] == 1

    def test_len(self, dataset: CropHarvest) -> None:
        assert len(dataset) == 5

    def test_already_downloaded(self, dataset: CropHarvest, tmp_path: Path) -> None:
        CropHarvest(root=str(tmp_path), download=False)

    def test_downloaded_zipped(self, dataset: CropHarvest, tmp_path: Path) -> None:
        feature_path = os.path.join(tmp_path, 'features')
        shutil.rmtree(feature_path)
        CropHarvest(root=str(tmp_path), download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CropHarvest(str(tmp_path))

    def test_plot(self, dataset: CropHarvest) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_mock_missing_module(
        self, dataset: CropHarvest, tmp_path: Path, mock_missing_module: None
    ) -> None:
        with pytest.raises(
            ImportError,
            match='h5py is not installed and is required to use this dataset',
        ):
            CropHarvest(root=str(tmp_path), download=True)[0]
