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
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import (
    NLCD,
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestNLCD:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> NLCD:
        monkeypatch.setattr(torchgeo.datasets.nlcd, 'download_url', download_url)

        md5s = {
            2011: '99546a3b89a0dddbe4e28e661c79984e',
            2019: 'a4008746f15720b8908ddd357a75fded',
        }
        monkeypatch.setattr(NLCD, 'md5s', md5s)

        url = os.path.join(
            'tests', 'data', 'nlcd', 'nlcd_{}_land_cover_l48_20210604.zip'
        )
        monkeypatch.setattr(NLCD, 'url', url)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity()
        return NLCD(
            root,
            transforms=transforms,
            download=True,
            checksum=True,
            years=[2011, 2019],
        )

    def test_getitem(self, dataset: NLCD) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_classes(self) -> None:
        root = os.path.join('tests', 'data', 'nlcd')
        classes = list(NLCD.cmap.keys())[:5]
        ds = NLCD(root, years=[2019], classes=classes)
        sample = ds[ds.bounds]
        mask = sample['mask']
        assert mask.max() < len(classes)

    def test_and(self, dataset: NLCD) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: NLCD) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: NLCD) -> None:
        NLCD(dataset.paths, download=True, years=[2019])

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join(
            'tests', 'data', 'nlcd', 'nlcd_2019_land_cover_l48_20210604.zip'
        )
        root = str(tmp_path)
        shutil.copy(pathname, root)
        NLCD(root, years=[2019])

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match='NLCD data product only exists for the following years:',
        ):
            NLCD(str(tmp_path), years=[1996])

    def test_invalid_classes(self) -> None:
        with pytest.raises(AssertionError):
            NLCD(classes=[-1])

        with pytest.raises(AssertionError):
            NLCD(classes=[11])

    def test_plot(self, dataset: NLCD) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: NLCD) -> None:
        query = dataset.bounds
        x = dataset[query]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            NLCD(str(tmp_path))

    def test_invalid_query(self, dataset: NLCD) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]
