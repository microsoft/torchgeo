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
from torchgeo.datasets import NLCD, BoundingBox, IntersectionDataset, UnionDataset


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestNLCD:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> NLCD:
        monkeypatch.setattr(torchgeo.datasets.nlcd, "download_url", download_url)

        md5s = {2019: "8dff19f58a6253d782c97f7981e26357"}
        monkeypatch.setattr(NLCD, "md5s", md5s)

        urls = {
            2019: os.path.join(
                "tests", "data", "nlcd", "nlcd_2019_land_cover_l48_20210604.zip"
            )
        }
        monkeypatch.setattr(NLCD, "urls", urls)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity()
        return NLCD(
            root, transforms=transforms, download=True, checksum=True, year=2019
        )

    def test_getitem(self, dataset: NLCD) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: NLCD) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: NLCD) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: NLCD) -> None:
        NLCD(root=dataset.root, download=True, year=2019)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join(
            "tests", "data", "nlcd", "nlcd_2019_land_cover_l48_20210604.zip"
        )
        root = str(tmp_path)
        shutil.copy(pathname, root)
        NLCD(root, year=2019)

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match="NLCD data product only exists for the following years:",
        ):
            NLCD(str(tmp_path), year=1996)

    def test_plot(self, dataset: NLCD) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: NLCD) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            NLCD(str(tmp_path))

    def test_invalid_query(self, dataset: NLCD) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
