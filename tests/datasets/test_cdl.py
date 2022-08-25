# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.datasets import CDL, BoundingBox, IntersectionDataset, UnionDataset


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestCDL:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> CDL:
        monkeypatch.setattr(torchgeo.datasets.cdl, "download_url", download_url)

        md5s = [
            (2021, "e929beb9c8e59fa1d7b7f82e64edaae1"),
            (2020, "e95c2d40ce0c261ed6ee0bd00b49e4b6"),
        ]
        monkeypatch.setattr(CDL, "md5s", md5s)
        url = os.path.join("tests", "data", "cdl", "{}_30m_cdls.zip")
        monkeypatch.setattr(CDL, "url", url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity()
        return CDL(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: CDL) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: CDL) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: CDL) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_full_year(self, dataset: CDL) -> None:
        bbox = dataset.bounds
        time = datetime(2021, 6, 1).timestamp()
        query = BoundingBox(bbox.minx, bbox.maxx, bbox.miny, bbox.maxy, time, time)
        next(dataset.index.intersection(tuple(query)))

    def test_already_extracted(self, dataset: CDL) -> None:
        CDL(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "cdl", "*_30m_cdls.zip")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        CDL(root)

    def test_plot(self, dataset: CDL) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: CDL) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            CDL(str(tmp_path))

    def test_invalid_query(self, dataset: CDL) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
