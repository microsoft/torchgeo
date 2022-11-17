# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for Modis Dataset."""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, IntersectionDataset, Modis, UnionDataset


class TestModis:
    @pytest.fixture
    def dataset(self) -> Modis:
        root = os.path.join("tests", "data", "modis")
        return Modis(root)

    def test_getitem(self, dataset: Modis) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_and(self, dataset: Modis) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Modis) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No Modis data was found in "):
            Modis(str(tmp_path))

    def test_plot(self, dataset: Modis) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_wrong_bands(self, dataset: Modis) -> None:
        bands = ["B02"]
        ds = Modis(root=dataset.root, res=dataset.res, bands=bands)
        x = dataset[dataset.bounds]
        with pytest.raises(
            ValueError, match="Dataset doesn't contain some of the RGB bands"
        ):
            ds.plot(x)

    def test_invalid_query(self, dataset: Modis) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
