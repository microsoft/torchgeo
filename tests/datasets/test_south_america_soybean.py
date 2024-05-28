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
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    SouthAmericaSoybean,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestSouthAmericaSoybean:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> SouthAmericaSoybean:
        monkeypatch.setattr(
            torchgeo.datasets.south_america_soybean, 'download_url', download_url
        )
        transforms = nn.Identity()
        url = os.path.join(
            'tests', 'data', 'south_america_soybean', 'SouthAmerica_Soybean_{}.tif'
        )

        monkeypatch.setattr(SouthAmericaSoybean, 'url', url)
        root = str(tmp_path)
        return SouthAmericaSoybean(
            paths=root,
            years=[2002, 2021],
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: SouthAmericaSoybean) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: SouthAmericaSoybean) -> None:
        assert len(dataset) == 2

    def test_and(self, dataset: SouthAmericaSoybean) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: SouthAmericaSoybean) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: SouthAmericaSoybean) -> None:
        SouthAmericaSoybean(dataset.paths, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join(
            'tests', 'data', 'south_america_soybean', 'SouthAmerica_Soybean_2002.tif'
        )
        root = str(tmp_path)
        shutil.copy(pathname, root)
        SouthAmericaSoybean(root)

    def test_plot(self, dataset: SouthAmericaSoybean) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: SouthAmericaSoybean) -> None:
        query = dataset.bounds
        x = dataset[query]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SouthAmericaSoybean(str(tmp_path))

    def test_invalid_query(self, dataset: SouthAmericaSoybean) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]
