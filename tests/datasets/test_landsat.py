# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pyproj import CRS

from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    Landsat8,
    RGBBandsMissingError,
    UnionDataset,
)


class TestLandsat8:
    @pytest.fixture(
        params=[
            ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
            ['SR_B4', 'SR_B3', 'SR_B2', 'SR_QA_AEROSOL'],
        ]
    )
    def dataset(self, request: SubRequest) -> Landsat8:
        root = os.path.join('tests', 'data', 'landsat8')
        bands = request.param
        transforms = nn.Identity()
        return Landsat8(root, bands=bands, transforms=transforms)

    def test_separate_files(self, dataset: Landsat8) -> None:
        assert dataset.index.count(dataset.index.bounds) == 1

    def test_getitem(self, dataset: Landsat8) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: Landsat8) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: Landsat8) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Landsat8) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: Landsat8) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_wrong_bands(self, dataset: Landsat8) -> None:
        bands = ('SR_B1',)
        ds = Landsat8(dataset.paths, bands=bands)
        x = dataset[dataset.bounds]
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds.plot(x)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Landsat8(tmp_path)

    def test_invalid_query(self, dataset: Landsat8) -> None:
        query = BoundingBox(0, 0, 0, 0, pd.Timestamp.min, pd.Timestamp.min)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]
