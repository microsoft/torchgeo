# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import BoundingBox, Esri2020, IntersectionDataset, UnionDataset


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestEsri2020:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> Esri2020:
        monkeypatch.setattr(torchgeo.datasets.esri2020, "download_url", download_url)
        zipfile = "io-lulc-model-001-v01-composite-v03-supercell-v02-clip-v01.zip"
        monkeypatch.setattr(Esri2020, "zipfile", zipfile)

        md5 = "34aec55538694171c7b605b0cc0d0138"
        monkeypatch.setattr(Esri2020, "md5", md5)
        url = os.path.join(
            "tests",
            "data",
            "esri2020",
            "io-lulc-model-001-v01-composite-v03-supercell-v02-clip-v01.zip",
        )
        monkeypatch.setattr(Esri2020, "url", url)
        root = str(tmp_path)
        transforms = nn.Identity()
        return Esri2020(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: Esri2020) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_already_extracted(self, dataset: Esri2020) -> None:
        Esri2020(root=dataset.root, download=True)

    def test_not_extracted(self, tmp_path: Path) -> None:
        url = os.path.join(
            "tests",
            "data",
            "esri2020",
            "io-lulc-model-001-v01-composite-v03-supercell-v02-clip-v01.zip",
        )
        shutil.copy(url, tmp_path)
        Esri2020(root=str(tmp_path))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            Esri2020(str(tmp_path), checksum=True)

    def test_and(self, dataset: Esri2020) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Esri2020) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: Esri2020) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: Esri2020) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_url(self) -> None:
        ds = Esri2020(os.path.join("tests", "data", "esri2020"))
        assert "ai4edataeuwest.blob.core.windows.net" in ds.url

    def test_invalid_query(self, dataset: Esri2020) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
