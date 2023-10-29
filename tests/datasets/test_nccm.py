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
from torchgeo.datasets import NCCM, BoundingBox, IntersectionDataset, UnionDataset


def download_url(url: str, root: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)

class TestNCCM:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> NCCM:
        
        monkeypatch.setattr(torchgeo.datasets.nccm, "download_url", download_url)
    
        md5s = {
            "main": "eae952f1b346d7e649d027e8139a76f5"
        }

        monkeypatch.setattr(NCCM, "md5s", md5s)

        url = os.path.join(
            "tests", "data", "nccm", "13090442.zip" 
        )

        monkeypatch.setattr(NCCM, "url", url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity()

        return NCCM(
            root,
            transforms=transforms,
            download=True,
            checksum=True,
            years=[2017,2019],
        )

    def test_getitem(self, dataset: NCCM) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_classes(self) -> None:
        root = os.path.join("tests", "data", "nccm")
        classes = list(NCCM.cmap.keys())[:5]
        ds = NCCM(root, years=[2019], classes=classes)
        sample = ds[ds.bounds]
        mask = sample["mask"]
        assert mask.max() < len(classes)
    
    def test_and(self, dataset: NCCM) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: NCCM) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    # def test_already_extracted(self, dataset: NCCM) -> None:
    #     NCCM(dataset.paths, download=True, years=[2019])
    
    # def test_already_downloaded(self, tmp_path: Path) -> None:
    #     pathname = os.path.join(
    #         "tests", "data", "nccm", "CDL2017_clip.zip"
    #     )
    #     root = str(tmp_path)
    #     shutil.copy(pathname, root)
    #     NCCM(root, years=[2019])

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match="NCCM data product only exists for the following years:",
        ):
            NCCM(str(tmp_path), years=[1996])

    def test_invalid_classes(self) -> None:
        with pytest.raises(AssertionError):
            NCCM(classes=[-1])

        with pytest.raises(AssertionError):
            NCCM(classes=[11])

    # def test_plot(self, dataset: NCCM) -> None:
    #     query = dataset.bounds
    #     x = dataset[query]
    #     dataset.plot(x, suptitle="Test")
    #     plt.close()

    # def test_plot_prediction(self, dataset: NCCM) -> None:
    #     query = dataset.bounds
    #     x = dataset[query]
    #     x["prediction"] = x["mask"].clone()
    #     dataset.plot(x, suptitle="Prediction")
    #     plt.close()

    # def test_not_downloaded(self, tmp_path: Path) -> None:
    #     with pytest.raises(RuntimeError, match="Dataset not found"):
    #         NCCM(str(tmp_path))

    # def test_invalid_query(self, dataset: NCCM) -> None:
    #     query = BoundingBox(0, 0, 0, 0, 0, 0)
    #     with pytest.raises(
    #         IndexError, match="query: .* not found in index with bounds:"
    #     ):
    #         dataset[query]
    


    

    
