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
from pytest import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    IOBench,
    RGBBandsMissingError,
    UnionDataset,
)


class TestIOBench:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> IOBench:
        md5 = 'e82398add7c35896a31c4398c608ef83'
        url = os.path.join('tests', 'data', 'iobench', '{}.tar.gz')
        monkeypatch.setattr(IOBench, 'url', url)
        monkeypatch.setitem(IOBench.md5s, 'preprocessed', md5)
        root = tmp_path
        transforms = nn.Identity()
        return IOBench(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: IOBench) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: IOBench) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: IOBench) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: IOBench) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: IOBench) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_already_extracted(self, dataset: IOBench) -> None:
        IOBench(dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'iobench', '*.tar.gz')
        root = tmp_path
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        IOBench(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            IOBench(tmp_path)

    def test_invalid_query(self, dataset: IOBench) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]

    def test_rgb_bands_absent_plot(self, dataset: IOBench) -> None:
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            print(dataset.root)
            ds = IOBench(dataset.root, bands=['SR_B1', 'SR_B2', 'SR_B3'])
            x = ds[ds.bounds]
            ds.plot(x, suptitle='Test')
            plt.close()
