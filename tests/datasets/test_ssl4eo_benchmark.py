# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from itertools import product
from pathlib import Path
from typing import Literal

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

        transforms = nn.Identity()
        return SSL4EOLBenchmark(
            root=root,
            sensor=sensor,
            product=product,
            split=split,
            transforms=transforms,
            download=True,
        )

    def test_getitem(self, dataset: SSL4EOLBenchmark) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: SSL4EOLBenchmark) -> None:
        assert len(dataset) == 3

    @pytest.mark.parametrize('product,base_class', [('nlcd', NLCD), ('cdl', CDL)])
    def test_classes(
        self, product: Literal['nlcd', 'cdl'], base_class: RasterDataset
    ) -> None:
        root = os.path.join('tests', 'data', 'ssl4eo_benchmark_landsat')
        classes = list(base_class.cmap.keys())[:5]
        ds = SSL4EOLBenchmark(root, product=product, classes=classes)
        sample = ds[0]
        mask = sample['mask']
        assert mask.max() < len(classes)

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
