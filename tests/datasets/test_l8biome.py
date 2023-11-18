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

import torchgeo.datasets.utils
from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    L8Biome,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestL8Biome:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> L8Biome:
        monkeypatch.setattr(torchgeo.datasets.l8biome, "download_url", download_url)
        md5s = {
            "barren": "29c9910adbc89677389f210226fb163d",
            "forest": "b7dbb82fb2c22cbb03389d8828d73713",
        }

        url = os.path.join("tests", "data", "l8biome", "{}.tar.gz")
        monkeypatch.setattr(L8Biome, "url", url)
        monkeypatch.setattr(L8Biome, "md5s", md5s)
        root = str(tmp_path)
        transforms = nn.Identity()
        return L8Biome(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: L8Biome) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: L8Biome) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: L8Biome) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: L8Biome) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_already_extracted(self, dataset: L8Biome) -> None:
        L8Biome(dataset.paths, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "l8biome", "*.tar.gz")
        root = str(tmp_path)
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        L8Biome(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            L8Biome(str(tmp_path))

    def test_plot_prediction(self, dataset: L8Biome) -> None:
        x = dataset[dataset.bounds]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_invalid_query(self, dataset: L8Biome) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_rgb_bands_absent_plot(self, dataset: L8Biome) -> None:
        with pytest.raises(
            ValueError, match="Dataset doesn't contain some of the RGB bands"
        ):
            ds = L8Biome(dataset.paths, bands=["B1", "B2", "B5"])
            x = ds[ds.bounds]
            ds.plot(x, suptitle="Test")
            plt.close()
