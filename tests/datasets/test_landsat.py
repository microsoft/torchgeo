# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Generator

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, IntersectionDataset, Landsat8, UnionDataset


class TestLandsat8:
    @pytest.fixture
    def dataset(self, monkeypatch: Generator[MonkeyPatch, None, None]) -> Landsat8:
        monkeypatch.setattr(  # type: ignore[attr-defined]
            plt, "show", lambda *args: None
        )
        root = os.path.join("tests", "data", "landsat8")
        bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        transforms = nn.Identity()  # type: ignore[attr-defined]
        return Landsat8(root, bands=bands, transforms=transforms)

    def test_separate_files(self, dataset: Landsat8) -> None:
        assert dataset.index.count(dataset.index.bounds) == 1

    def test_getitem(self, dataset: Landsat8) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_and(self, dataset: Landsat8) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Landsat8) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: Landsat8) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x["image"])

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No Landsat8 data was found in "):
            Landsat8(str(tmp_path))

    def test_invalid_query(self, dataset: Landsat8) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
