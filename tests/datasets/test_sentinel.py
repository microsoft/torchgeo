# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, IntersectionDataset, Sentinel2, UnionDataset


class TestSentinel2:
    @pytest.fixture
    def dataset(self) -> Sentinel2:
        root = os.path.join("tests", "data", "sentinel2")
        res = 10
        bands = ["B02", "B03", "B04", "B08"]
        transforms = nn.Identity()
        return Sentinel2(root, res=res, bands=bands, transforms=transforms)

    def test_separate_files(self, dataset: Sentinel2) -> None:
        assert dataset.index.count(dataset.index.bounds) == 2

    def test_getitem(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_and(self, dataset: Sentinel2) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Sentinel2) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No Sentinel2 data was found in "):
            Sentinel2(str(tmp_path))

    def test_plot(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_wrong_bands(self, dataset: Sentinel2) -> None:
        bands = ["B02"]
        ds = Sentinel2(root=dataset.root, res=dataset.res, bands=bands)
        x = dataset[dataset.bounds]
        with pytest.raises(
            ValueError, match="Dataset doesn't contain some of the RGB bands"
        ):
            ds.plot(x)

    def test_invalid_query(self, dataset: Sentinel2) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
