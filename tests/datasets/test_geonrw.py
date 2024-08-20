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
from torchgeo.datasets import DatasetNotFoundError, GeoNRW


def download_url(url: str, root: str | Path, *args: str) -> None:
    shutil.copy(url, root)


class TestGeoNRW:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> GeoNRW:
        monkeypatch.setattr(torchgeo.datasets.utils, 'download_url', download_url)
        md5 = '6ffc014d4b345bba3076e8d76ab481fa'
        monkeypatch.setattr(GeoNRW, 'md5', md5)
        url = os.path.join('tests', 'data', 'geonrw', 'nrw_dataset.tar.gz')
        monkeypatch.setattr(GeoNRW, 'url', url)
        monkeypatch.setattr(GeoNRW, 'train_list', ['aachen', 'bergisch', 'bielefeld'])
        monkeypatch.setattr(GeoNRW, 'test_list', ['duesseldorf'])
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return GeoNRW(root, split, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: GeoNRW) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].shape[-2:] == x['mask'].shape[-2:]

    def test_len(self, dataset: GeoNRW) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 6
        else:
            assert len(dataset) == 2

    def test_already_downloaded(self, dataset: GeoNRW) -> None:
        GeoNRW(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        filename = 'nrw_dataset.tar.gz'
        dir = os.path.join('tests', 'data', 'geonrw')
        shutil.copyfile(
            os.path.join(dir, filename), os.path.join(str(tmp_path), filename)
        )
        GeoNRW(root=str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            GeoNRW(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            GeoNRW(tmp_path)

    def test_plot(self, dataset: GeoNRW) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = torch.clone(sample['mask'])
        dataset.plot(sample, suptitle='Prediction')
        plt.close()
