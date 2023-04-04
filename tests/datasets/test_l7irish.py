# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from datetime import datetime
from pathlib import Path

import pytest
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
# from torch.utils.data import DataLoader
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

import torchgeo.datasets.utils
from torchgeo.samplers import RandomGeoSampler
from torchgeo.datasets import L7Irish, BoundingBox, IntersectionDataset, UnionDataset


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)

# ds = l7irish('data/l7irish', res=300)
# sampler = RandomGeoSampler(ds, size=5000)
# dl = DataLoader(ds, sampler=sampler)
# bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
# sample = ds[bbox]
# ds.plot(sample)
# plt.show()

class TestL7Irish:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> L7Irish:
        monkeypatch.setattr(torchgeo.datasets.l7irish, "download_url", download_url)
        md5s = {
            "austral": "65daad5bf33ecbe35a35ddf4b220d549",
        }
        monkeypatch.setattr(L7Irish, "md5s", md5s)
        url = os.path.join("tests", "data", "l7irish", "{}.tar.gz")
        monkeypatch.setattr(L7Irish, "url", url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity()
        return L7Irish(root, transforms=transforms, download=True, checksum=True)
    
    # def dataset(self) -> L7Irish:
    #     root = os.path.join("tests", "data", "l7irish")
    #     transforms = nn.Identity()
    #     return L7Irish(root, transforms=transforms)

    def test_getitem(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        # assert isinstance(x["mask"], torch.Tensor)
        

    def test_and(self, dataset: L7Irish) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: L7Irish) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: L7Irish) -> None:
        L7Irish(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "l7irish", "*.tar.gz")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        L7Irish(root)

    def test_plot(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: L7Irish) -> None: 
        x = dataset[dataset.bounds]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found"):
            L7Irish(str(tmp_path))

    def test_invalid_query(self, dataset: L7Irish) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]