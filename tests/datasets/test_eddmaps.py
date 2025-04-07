# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    EDDMapS,
    IntersectionDataset,
    UnionDataset,
)


class TestEDDMapS:
    @pytest.fixture(scope='class')
    def dataset(self) -> EDDMapS:
        root = os.path.join('tests', 'data', 'eddmaps')
        return EDDMapS(root)

    def test_getitem(self, dataset: EDDMapS) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)

    def test_len(self, dataset: EDDMapS) -> None:
        assert len(dataset) == 3

    def test_and(self, dataset: EDDMapS) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: EDDMapS) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EDDMapS(tmp_path)

    def test_invalid_query(self, dataset: EDDMapS) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]

    def test_plot(self, dataset: EDDMapS) -> None:
        # Test with default parameters
        fig, ax = dataset.plot()
        assert fig is not None
        assert ax is not None
        plt.close(fig)  # Clean up

        # Test with specific BoundingBox
        query = BoundingBox(
            minx=-88.0,
            maxx=-87.0,
            miny=41.0,
            maxy=42.0,
            mint=0,
            maxt=sys.maxsize,  # Full time range
        )
        fig, ax = dataset.plot(query=query)
        assert fig is not None
        assert ax is not None
        plt.close(fig)  # Clean up
