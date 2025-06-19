# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from torchgeo.datasets import (
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
        assert len(dataset) == 2

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
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_plot(self, dataset: EDDMapS) -> None:
        sample = dataset[dataset.bounds]
        fig = dataset.plot(sample, suptitle='test')
        assert isinstance(fig, Figure)
        plt.close()
