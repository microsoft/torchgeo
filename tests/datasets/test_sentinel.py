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
    RGBBandsMissingError,
    Sentinel1,
    Sentinel2,
    UnionDataset,
)


class TestSentinel1:
    @pytest.fixture(
        params=[
            # Only horizontal or vertical receive
            ['HH'],
            ['HV'],
            ['VV'],
            ['VH'],
            # Both horizontal and vertical receive
            ['HH', 'HV'],
            ['HV', 'HH'],
            ['VV', 'VH'],
            ['VH', 'VV'],
        ]
    )
    def dataset(self, request: SubRequest) -> Sentinel1:
        root = os.path.join('tests', 'data', 'sentinel1')
        bands = request.param
        transforms = nn.Identity()
        return Sentinel1(root, bands=bands, transforms=transforms)

    def test_separate_files(self, dataset: Sentinel1) -> None:
        assert dataset.index.count(dataset.index.bounds) == 1

    def test_getitem(self, dataset: Sentinel1) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: Sentinel1) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: Sentinel1) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Sentinel1) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Sentinel1(tmp_path)

    def test_empty_bands(self) -> None:
        with pytest.raises(AssertionError, match="'bands' cannot be an empty list"):
            Sentinel1(bands=[])

    @pytest.mark.parametrize('bands', [['HH', 'HH'], ['HH', 'HV', 'HH']])
    def test_duplicate_bands(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="'bands' contains duplicate bands"):
            Sentinel1(bands=bands)

    @pytest.mark.parametrize('bands', [['HH_HV'], ['HH', 'HV', 'HH_HV']])
    def test_invalid_bands(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="invalid band 'HH_HV'"):
            Sentinel1(bands=bands)

    @pytest.mark.parametrize(
        'bands', [['HH', 'VV'], ['HH', 'VH'], ['VV', 'HV'], ['HH', 'HV', 'VV', 'VH']]
    )
    def test_dual_transmit(self, bands: list[str]) -> None:
        with pytest.raises(AssertionError, match="'bands' cannot contain both "):
            Sentinel1(bands=bands)

    def test_invalid_query(self, dataset: Sentinel1) -> None:
        query = BoundingBox(-1, -1, -1, -1, pd.Timestamp.min, pd.Timestamp.min)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]


class TestSentinel2:
    @pytest.fixture
    def dataset(self) -> Sentinel2:
        root = os.path.join('tests', 'data', 'sentinel2')
        res = (10.0, 10.0)
        bands = ['B02', 'B03', 'B04', 'B08']
        transforms = nn.Identity()
        return Sentinel2(root, res=res, bands=bands, transforms=transforms)

    def test_separate_files(self, dataset: Sentinel2) -> None:
        assert dataset.index.count(dataset.index.bounds) == 4

    def test_getitem(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: Sentinel2) -> None:
        assert len(dataset) == 4

    def test_and(self, dataset: Sentinel2) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: Sentinel2) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            Sentinel2(tmp_path)

    def test_plot(self, dataset: Sentinel2) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_wrong_bands(self, dataset: Sentinel2) -> None:
        bands = ['B02']
        ds = Sentinel2(dataset.paths, res=dataset.res, bands=bands)
        x = dataset[dataset.bounds]
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds.plot(x)

    def test_invalid_query(self, dataset: Sentinel2) -> None:
        query = BoundingBox(0, 0, 0, 0, pd.Timestamp.min, pd.Timestamp.min)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]

    def test_float_res(self, dataset: Sentinel2) -> None:
        Sentinel2(dataset.paths, res=10.0, bands=dataset.bands)
