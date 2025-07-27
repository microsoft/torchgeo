# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pyproj import CRS

from torchgeo.datasets import (
    Airphen,
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    UnionDataset,
)


class TestAirphen:
    @pytest.fixture
    def dataset(self) -> Airphen:
        paths = os.path.join('tests', 'data', 'airphen')
        bands = ['B1', 'B3', 'B4']
        transforms = nn.Identity()
        return Airphen(paths, bands=bands, transforms=transforms)

    def test_len(self, dataset: Airphen) -> None:
        assert len(dataset) == 1

    def test_getitem(self, dataset: Airphen) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)

    def test_and(self, dataset: Airphen) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Airphen) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: Airphen) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Airphen(tmp_path)

    def test_invalid_query(self, dataset: Airphen) -> None:
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_plot_wrong_bands(self, dataset: Airphen) -> None:
        bands = ('B1', 'B2', 'B3')
        ds = Airphen(dataset.paths, bands=bands)
        x = dataset[dataset.bounds]
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds.plot(x)
