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
        # need to create new md5s?
        md5s = {
            "austral": "dbb6b5628f50861b9b89f548d25a925f",
            "boreal": "cecc72de09aacde4c4f8d7f0cf0d3f6f",
            "mid_latitude_north": "0f8382ca6554fb7cf9aff42226a14f9d",
            "mid_latitude_south": "b17cf6d023f752c533211fdb742f296b",
            "polar_north": "73923dcaf1b9b79bad82de1aa0740d1e",
            "polar_south": "3bc9f4c6f8955b10b4d55d23e0ab2da7",
            "subtropical_north": "f8f039970256902e6e9ebd6747589294",
            "subtropical_south": "8346d73a983396c5d41b577c3a94bc26",
            "tropical": "abe19b22b5d031e6b609cc7207706c3d",
        }
        
        monkeypatch.setattr(L7Irish, "md5s", md5s)
        url = os.path.join("tests", "data", "l7irish", "{}.tar.gz")
        monkeypatch.setattr(L7Irish, "url", url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity() # why does it need nn?
        return L7Irish(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS) # this is different from landcoverai
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: L7Irish) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: L7Irish) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_full_year(self, dataset: L7Irish) -> None:
        bbox = dataset.bounds
        time = datetime(2001, 11, 12).timestamp() # randomly select one time
        query = BoundingBox(bbox.minx, bbox.maxx, bbox.miny, bbox.maxy, time, time)
        next(dataset.index.intersection(tuple(query)))

    def test_already_extracted(self, dataset: L7Irish) -> None:
        L7Irish(root=dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join("tests", "data", "l7irish", "*.tar.gz")
        root = str(tmp_path)
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        L7Irish(root)

    def test_plot(self, dataset: L7Irish) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: L7Irish) -> None: 
        query = dataset.bounds # same as cdl, slightly different from landcoverai
        x = dataset[query]
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


#help(BoundingBox)
#help(RAndomGeoSampler)

# for batch in dl:
# 	samples = unbind_samples(batch)
# 	sample = sample[0]
# 	ds.plot(sample)

# bbox = ds.index.bounds

# ds[bbox]

# ds.plot(sample)

# make new file for test l7irish in test directory
# don't need params
# monkeypatch: make fake data and checksums and use it for testing
# create fake data: /tests/data/ to see example, be careful about different resolutions,
# black
# isort
# flake8
# put pull request @...