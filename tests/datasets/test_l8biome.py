# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import BoundingBox, IntersectionDataset, L8Biome, UnionDataset

class TestL8Biome:
    
    ## Still working on tests
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> L8Biome:
        root = os.path.join("tests", "data", "l8biome")
        bands = request.param
        transforms = nn.Identity()
        return L8Biome(root, transforms=transforms)

    def test_separate_files(self, dataset: L8Biome) -> None:
        assert dataset.index.count(dataset.index.bounds) == 1

    # def test_getitem(self, dataset: L8Biome) -> None:
    #     x = dataset[dataset.bounds]
    #     assert isinstance(x, dict)
    #     assert isinstance(x["crs"], CRS)
    #     assert isinstance(x["image"], torch.Tensor)

    def test_and(self, dataset: L8Biome) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: L8Biome) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    # def test_plot(self, dataset: L8Biome) -> None:
    #     x = dataset[dataset.bounds]
    #     dataset.plot(x, suptitle="Test")
    #     plt.close()

    # def test_plot_wrong_bands(self, dataset: L8Biome) -> None:
    #     bands = ("SR_B1",)
    #     ds = L8Biome(root=dataset.root, bands=bands)
    #     x = dataset[dataset.bounds]
    #     with pytest.raises(
    #         ValueError, match="Dataset doesn't contain some of the RGB bands"
    #     ):
    #         ds.plot(x)

    # def test_no_data(self, tmp_path: Path) -> None:
    #     with pytest.raises(FileNotFoundError, match="No L8Biome data was found in "):
    #         L8Biome(str(tmp_path))

    # def test_invalid_query(self, dataset: L8Biome) -> None:
    #     query = BoundingBox(0, 0, 0, 0, 0, 0)
    #     with pytest.raises(
    #         IndexError, match="query: .* not found in index with bounds:"
    #     ):
    #         dataset[query]