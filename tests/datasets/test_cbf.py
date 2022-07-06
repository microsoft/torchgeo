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
from torchgeo.datasets import (
    BoundingBox,
    CanadianBuildingFootprints,
    IntersectionDataset,
    UnionDataset,
)


def download_url(url: str, root: str, *args: str) -> None:
    shutil.copy(url, root)


class TestCanadianBuildingFootprints:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> CanadianBuildingFootprints:
        monkeypatch.setattr(torchgeo.datasets.utils, "download_url", download_url)
        monkeypatch.setattr(
            CanadianBuildingFootprints, "provinces_territories", ["Alberta"]
        )
        monkeypatch.setattr(
            CanadianBuildingFootprints, "md5s", ["25091d1f051baa30d8f2026545cfb696"]
        )
        url = os.path.join("tests", "data", "cbf") + os.sep
        monkeypatch.setattr(CanadianBuildingFootprints, "url", url)
        monkeypatch.setattr(plt, "show", lambda *args: None)
        root = str(tmp_path)
        transforms = nn.Identity()
        return CanadianBuildingFootprints(
            root, res=0.1, transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: CanadianBuildingFootprints) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: CanadianBuildingFootprints) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: CanadianBuildingFootprints) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: CanadianBuildingFootprints) -> None:
        CanadianBuildingFootprints(root=dataset.root, download=True)

    def test_plot(self, dataset: CanadianBuildingFootprints) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")

    def test_plot_prediction(self, dataset: CanadianBuildingFootprints) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="Dataset not found or corrupted."):
            CanadianBuildingFootprints(str(tmp_path))

    def test_invalid_query(self, dataset: CanadianBuildingFootprints) -> None:
        query = BoundingBox(2, 2, 2, 2, 2, 2)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
