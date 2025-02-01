# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    CDL,
    NLCD,
    DatasetNotFoundError,
    RasterDataset,
    SSL4EOLBenchmark,
)


class TestSSL4EOLBenchmark:
    @pytest.fixture(
        params=product(
            ['tm_toa', 'etm_toa', 'etm_sr', 'oli_tirs_toa', 'oli_sr'],
            ['cdl', 'nlcd'],
            ['train', 'val', 'test'],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SSL4EOLBenchmark:
        root = tmp_path
        url = os.path.join('tests', 'data', 'ssl4eo_benchmark_landsat', '{}.tar.gz')
        monkeypatch.setattr(SSL4EOLBenchmark, 'url', url)

        sensor, product, split = request.param
        monkeypatch.setattr(
            SSL4EOLBenchmark, 'split_percentages', [1 / 3, 1 / 3, 1 / 3]
        )

        img_md5s = {
            'tm_toa': '81e3c0701057957f5d323483c3b4a871',
            'etm_sr': 'b38eac2744c36cc2929c11d141e21b2c',
            'etm_toa': '80172f2621eb4d5633b90f4344ad1d3d',
            'oli_tirs_toa': '5df398ecae45c86005b489dbe657f4bf',
            'oli_sr': 'd27ac929c2c1c84925077f35bdbebf5f',
        }
        monkeypatch.setattr(SSL4EOLBenchmark, 'img_md5s', img_md5s)

        mask_md5s = {
            'tm': {
                'cdl': '4b825e55a48c58f1ae5b3893987dca45',
                'nlcd': '2df58c68c636f941f938618214a9118c',
            },
            'etm': {
                'cdl': '4e854aadf8309a102d9fbf322f52a122',
                'nlcd': 'ebeae394bef0dace53ba83ba6ac3943c',
            },
            'oli': {
                'cdl': '33ab6ba3b3dc3ad15b34264392883bbf',
                'nlcd': 'dd044814e5df845ff8583d9ce0883c0f',
            },
        }
        monkeypatch.setattr(SSL4EOLBenchmark, 'mask_md5s', mask_md5s)

        transforms = nn.Identity()
        return SSL4EOLBenchmark(
            root=root,
            sensor=sensor,
            product=product,
            split=split,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: SSL4EOLBenchmark) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: SSL4EOLBenchmark) -> None:
        assert len(dataset) == 3

    @pytest.mark.parametrize('product,base_class', [('nlcd', NLCD), ('cdl', CDL)])
    def test_classes(self, product: str, base_class: RasterDataset) -> None:
        root = os.path.join('tests', 'data', 'ssl4eo_benchmark_landsat')
        classes = list(base_class.cmap.keys())[:5]
        ds = SSL4EOLBenchmark(root, product=product, classes=classes)
        sample = ds[0]
        mask = sample['mask']
        assert mask.max() < len(classes)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(split='foo')

    def test_invalid_sensor(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(sensor='foo')

    def test_invalid_product(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(product='foo')

    def test_invalid_classes(self) -> None:
        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(classes=[-1])

        with pytest.raises(AssertionError):
            SSL4EOLBenchmark(classes=[11])

    def test_add(self, dataset: SSL4EOLBenchmark) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: SSL4EOLBenchmark) -> None:
        SSL4EOLBenchmark(
            root=dataset.root,
            sensor=dataset.sensor,
            product=dataset.product,
            download=True,
        )

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'ssl4eo_benchmark_landsat', '*.tar.gz')
        root = tmp_path
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        SSL4EOLBenchmark(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SSL4EOLBenchmark(tmp_path)

    def test_plot(self, dataset: SSL4EOLBenchmark) -> None:
        sample = dataset[0]
        dataset.plot(sample, suptitle='Test')
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample)
        plt.close()
