# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from pyproj import CRS

from torchgeo.datasets import (
    DatasetNotFoundError,
    GlobalBuildingMap,
    IntersectionDataset,
    UnionDataset,
)


class TestGlobalBuildingMap:
    @pytest.fixture
    def dataset(self) -> GlobalBuildingMap:
        paths = os.path.join('tests', 'data', 'gbm')
        return GlobalBuildingMap(paths)

    def test_getitem(self, dataset: GlobalBuildingMap) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: GlobalBuildingMap) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: GlobalBuildingMap) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: GlobalBuildingMap) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: GlobalBuildingMap) -> None:
        sample = dataset[dataset.bounds]
        sample['prediction'] = sample['mask']
        dataset.plot(sample, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            GlobalBuildingMap(tmp_path)

    def test_invalid_query(self, dataset: GlobalBuildingMap) -> None:
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
