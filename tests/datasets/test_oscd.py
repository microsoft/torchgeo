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
    @pytest.fixture(
        params=zip([OSCD, OSCD100], [OSCD.all_bands, OSCD.rgb_bands], ['train', 'test'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> OSCD:
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
        cls, bands, split = request.param
        monkeypatch.setattr(cls, 'urls', urls)
        return cls(tmp_path, split, bands, transforms=nn.Identity(), download=True)

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
        type(dataset)(root=dataset.root, download=True)

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
        single_band_dataset = type(dataset)(root=dataset.root, bands=('B01',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            x = single_band_dataset[0].copy()
            single_band_dataset.plot(x, suptitle='Test')
