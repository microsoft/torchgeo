# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pyproj import CRS

from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    EnMAP,
    IntersectionDataset,
    RGBBandsMissingError,
    UnionDataset,
)


class TestEnMAP:
    @pytest.fixture
    def dataset(self) -> EnMAP:
        root = os.path.join('tests', 'data', 'enmap')
        transforms = nn.Identity()
        return EnMAP(root, transforms=transforms)

    def test_getitem(self, dataset: EnMAP) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: EnMAP) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: EnMAP) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: EnMAP) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: EnMAP) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_wrong_bands(self, dataset: EnMAP) -> None:
        bands = ('B1', 'B2', 'B3')
        ds = EnMAP(dataset.paths, bands=bands)
        x = dataset[dataset.bounds]
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds.plot(x)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EnMAP(tmp_path)

    def test_invalid_query(self, dataset: EnMAP) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]
