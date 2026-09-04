# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from torch import nn

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    PrestoEmbeddings,
    UnionDataset,
)


class TestPrestoEmbeddings:
    @pytest.fixture
    def dataset(self, test_data: Callable[[str], str]) -> PrestoEmbeddings:
        paths = test_data('presto')
        transforms = nn.Identity()
        return PrestoEmbeddings(paths, transforms=transforms)

    def test_len(self, dataset: PrestoEmbeddings) -> None:
        assert len(dataset) == 1

    def test_getitem(self, dataset: PrestoEmbeddings) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_and(self, dataset: PrestoEmbeddings) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: PrestoEmbeddings) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: PrestoEmbeddings) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            PrestoEmbeddings(tmp_path)

    def test_invalid_index(self, dataset: PrestoEmbeddings) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
