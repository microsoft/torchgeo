# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    SouthAfricaCropType,
    UnionDataset,
)


class TestSouthAfricaCropType:
    @pytest.fixture(params=[SouthAfricaCropType.s1_bands, SouthAfricaCropType.s2_bands])
    def dataset(self, request: SubRequest) -> SouthAfricaCropType:
        path = os.path.join("tests", "data", "south_africa_crop_type")
        bands = request.param
        transforms = nn.Identity()
        return SouthAfricaCropType(path, bands=bands, transforms=transforms)

    def test_getitem(self, dataset: SouthAfricaCropType) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["mask"], torch.Tensor)

    def test_and(self, dataset: SouthAfricaCropType) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: SouthAfricaCropType) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: SouthAfricaCropType) -> None:
        SouthAfricaCropType(paths=dataset.paths)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match="Dataset not found"):
            SouthAfricaCropType(str(tmp_path))

    def test_plot(self) -> None:
        path = os.path.join("tests", "data", "south_africa_crop_type")
        ds = SouthAfricaCropType(path)
        x = ds[ds.bounds]
        ds.plot(x, suptitle="Test")
        plt.close()

    def test_plot_prediction(self) -> None:
        path = os.path.join("tests", "data", "south_africa_crop_type")
        ds = SouthAfricaCropType(path)
        x = ds[ds.bounds]
        x["prediction"] = x["mask"].clone()
        ds.plot(x, suptitle="Prediction")
        plt.close()

    def test_invalid_query(self, dataset: SouthAfricaCropType) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_rgb_bands_absent_plot(self) -> None:
        path = os.path.join("tests", "data", "south_africa_crop_type")
        with pytest.raises(
            RGBBandsMissingError, match="Dataset does not contain some of the RGB bands"
        ):
            ds = SouthAfricaCropType(path, bands=SouthAfricaCropType.s1_bands)
            x = ds[ds.bounds]
            ds.plot(x, suptitle="Test")
            plt.close()
