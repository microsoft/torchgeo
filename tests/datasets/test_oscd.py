# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from matplotlib import pyplot as plt
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import OSCD, OSCD100, DatasetNotFoundError, RGBBandsMissingError


class TestOSCD:
    @pytest.fixture(params=zip([OSCD.all_bands, OSCD.rgb_bands], ['train', 'test']))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> OSCD:
        md5s = {
            'Onera Satellite Change Detection dataset - Images.zip': (
                'fb4e3f54c3a31fd3f21f98cad4ddfb74'
            ),
            'Onera Satellite Change Detection dataset - Train Labels.zip': (
                'ca526434a60e9abdf97d528dc29e9f13'
            ),
            'Onera Satellite Change Detection dataset - Test Labels.zip': (
                'ca0ba73ba66d06fa4903e269ef12eb50'
            ),
        }
        monkeypatch.setattr(OSCD, 'md5s', md5s)
        urls = {
            'Onera Satellite Change Detection dataset - Images.zip': os.path.join(
                'tests',
                'data',
                'oscd',
                'Onera Satellite Change Detection dataset - Images.zip',
            ),
            'Onera Satellite Change Detection dataset - Train Labels.zip': os.path.join(
                'tests',
                'data',
                'oscd',
                'Onera Satellite Change Detection dataset - Train Labels.zip',
            ),
            'Onera Satellite Change Detection dataset - Test Labels.zip': os.path.join(
                'tests',
                'data',
                'oscd',
                'Onera Satellite Change Detection dataset - Test Labels.zip',
            ),
        }
        monkeypatch.setattr(OSCD, 'urls', urls)

        bands, split = request.param
        root = tmp_path
        transforms = nn.Identity()
        return OSCD(
            root, split, bands, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: OSCD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].ndim == 4
        assert isinstance(x['mask'], torch.Tensor)
        assert x['mask'].ndim == 3

        if dataset.bands == OSCD.rgb_bands:
            assert x['image'].shape[1] == 3
        else:
            assert x['image'].shape[1] == 13

    def test_len(self, dataset: OSCD) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 4
        else:
            assert len(dataset) == 2

    def test_add(self, dataset: OSCD) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: OSCD) -> None:
        OSCD(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'oscd', '*Onera*.zip')
        root = tmp_path
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        OSCD(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OSCD(tmp_path)

    def test_plot(self, dataset: OSCD) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

    def test_failed_plot(self, dataset: OSCD) -> None:
        single_band_dataset = OSCD(root=dataset.root, bands=('B01',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            x = single_band_dataset[0].copy()
            single_band_dataset.plot(x, suptitle='Test')


class TestOSCD100:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> OSCD100:
        directory = os.path.join('tests', 'data', 'oscd', 'oscd100')
        urls = {
            'Onera Satellite Change Detection dataset - Images.zip': os.path.join(
                directory, 'Onera Satellite Change Detection dataset - Images.zip'
            ),
            'Onera Satellite Change Detection dataset - Train Labels.zip': os.path.join(
                directory, 'Onera Satellite Change Detection dataset - Train Labels.zip'
            ),
            'Onera Satellite Change Detection dataset - Val Labels.zip': os.path.join(
                directory, 'Onera Satellite Change Detection dataset - Val Labels.zip'
            ),
            'Onera Satellite Change Detection dataset - Test Labels.zip': os.path.join(
                directory, 'Onera Satellite Change Detection dataset - Test Labels.zip'
            ),
        }
        md5s = {
            'Onera Satellite Change Detection dataset - Images.zip': '4fbb84f1327f1c8eb9304ae79711f20a',
            'Onera Satellite Change Detection dataset - Train Labels.zip': '67f9244aea9201e76bac53b4227728d6',
            'Onera Satellite Change Detection dataset - Val Labels.zip': '42f5f268c698bf2f32766dda45f8011b',
            'Onera Satellite Change Detection dataset - Test Labels.zip': '75e462452c6e0c2c7b0f96ec8097b734',
        }
        monkeypatch.setattr(OSCD100, 'urls', urls)
        monkeypatch.setattr(OSCD100, 'md5s', md5s)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return OSCD100(root=root, split=split, transforms=transforms, download=True)

    def test_getitem(self, dataset: OSCD100) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].shape[1] == 13

    def test_len(self, dataset: OSCD100) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: OSCD100) -> None:
        OSCD100(root=dataset.root, download=True)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            OSCD100(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OSCD100(tmp_path)

    def test_plot(self, dataset: OSCD100) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Prediction')
        plt.close()
