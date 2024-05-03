# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from rasterio.crs import CRS

from torchgeo.datasets import (
    AgriFieldNet,
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    RGBBandsMissingError,
    UnionDataset,
)


class TestAgriFieldNet:
    @pytest.fixture
    def dataset(self) -> AgriFieldNet:
        path = os.path.join('tests', 'data', 'agrifieldnet')
        transforms = nn.Identity()
        return AgriFieldNet(paths=path, transforms=transforms)

    def test_getitem(self, dataset: AgriFieldNet) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_and(self, dataset: AgriFieldNet) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: AgriFieldNet) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: AgriFieldNet) -> None:
        AgriFieldNet(paths=dataset.paths)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            AgriFieldNet(str(tmp_path))

    def test_plot(self, dataset: AgriFieldNet) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: AgriFieldNet) -> None:
        x = dataset[dataset.bounds]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_query(self, dataset: AgriFieldNet) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]

    def test_rgb_bands_absent_plot(self, dataset: AgriFieldNet) -> None:
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds = AgriFieldNet(dataset.paths, bands=['B01', 'B02', 'B05'])
            x = ds[ds.bounds]
            ds.plot(x, suptitle='Test')
            plt.close()
