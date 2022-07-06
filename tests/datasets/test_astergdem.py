# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from rasterio.crs import CRS

from torchgeo.datasets import AsterGDEM, BoundingBox, IntersectionDataset, UnionDataset


class TestAsterGDEM:
    @pytest.fixture
    def dataset(self, tmp_path: Path) -> AsterGDEM:
        zipfile = os.path.join("tests", "data", "astergdem", "astergdem.zip")
        shutil.unpack_archive(zipfile, tmp_path, "zip")
        root = str(tmp_path)
        transforms = nn.Identity()
        return AsterGDEM(root, transforms=transforms)

    def test_datasetmissing(self, tmp_path: Path) -> None:
        shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)
        with pytest.raises(RuntimeError, match="Dataset not found in"):
            AsterGDEM(root=str(tmp_path))

    def test_getitem(self, dataset: AsterGDEM) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: AsterGDEM) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: AsterGDEM) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: AsterGDEM) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self, dataset: AsterGDEM) -> None:
        query = dataset.bounds
        x = dataset[query]
        x["prediction"] = x["mask"].clone()
        dataset.plot(x, suptitle="Prediction")
        plt.close()

    def test_invalid_query(self, dataset: AsterGDEM) -> None:
        query = BoundingBox(100, 100, 100, 100, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
