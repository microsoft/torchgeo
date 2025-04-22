# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from matplotlib.figure import Figure

from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    INaturalist,
    IntersectionDataset,
    UnionDataset,
)


class TestINaturalist:
    @pytest.fixture(scope='class')
    def dataset(self) -> INaturalist:
        root = os.path.join('tests', 'data', 'inaturalist')
        return INaturalist(root)

    def test_getitem(self, dataset: INaturalist) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)

    def test_len(self, dataset: INaturalist) -> None:
        assert len(dataset) == 3

    def test_and(self, dataset: INaturalist) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: INaturalist) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            INaturalist(tmp_path)

    def test_invalid_query(self, dataset: INaturalist) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]

    def test_plot(self, dataset: INaturalist) -> None:
        sample = dataset[dataset.bounds]
        fig = dataset.plot(sample, suptitle='test')
        assert isinstance(fig, Figure)
        plt.close()
