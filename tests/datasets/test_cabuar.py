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
from torchgeo.datasets import CaBuAr, DatasetNotFoundError

pytest.importorskip('h5py', minversion='3.6')


def download_url(
    url: str, root: str | Path, filename: str, *args: str, **kwargs: str
) -> None:
    shutil.copy(url, os.path.join(root, filename))


class TestCaBuAr:
    @pytest.fixture(
        params=zip([CaBuAr.all_bands, CaBuAr.rgb_bands], ['train', 'val', 'test'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> CaBuAr:
        monkeypatch.setattr(torchgeo.datasets.cabuar, 'download_url', download_url)
        data_dir = os.path.join('tests', 'data', 'cabuar')
        urls = (
            os.path.join(data_dir, '512x512.hdf5'),
            os.path.join(data_dir, 'chabud_test.h5'),
        )
        md5s = ('fd7d2f800562a5bb2c9f101ebb9104b2', '41ba3903e7d9db2d549c72261d6a6d53')
        monkeypatch.setattr(CaBuAr, 'urls', urls)
        monkeypatch.setattr(CaBuAr, 'md5s', md5s)
        bands, split = request.param
        root = tmp_path
        transforms = nn.Identity()
        return CaBuAr(
            root=root,
            split=split,
            bands=bands,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: CaBuAr) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

        # Image tests
        assert x['image'].ndim == 3

        if dataset.bands == CaBuAr.rgb_bands:
            assert x['image'].shape[0] == 2 * 3
        elif dataset.bands == CaBuAr.all_bands:
            assert x['image'].shape[0] == 2 * 12

        # Mask tests:
        assert x['mask'].ndim == 2

    def test_len(self, dataset: CaBuAr) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(self, dataset: CaBuAr) -> None:
        CaBuAr(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CaBuAr(tmp_path)

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            CaBuAr(bands=('OK', 'BK'))

    def test_plot(self, dataset: CaBuAr) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='prediction')
        plt.close()

    def test_plot_rgb(self, dataset: CaBuAr) -> None:
        dataset = CaBuAr(root=dataset.root, bands=('B02',))
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], suptitle='Single Band')

    def test_invalid_split(self, dataset: CaBuAr) -> None:
        with pytest.raises(AssertionError):
            CaBuAr(dataset.root, split='foo')
