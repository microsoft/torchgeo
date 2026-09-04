# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch

from torchgeo.datasets import (
    DatasetNotFoundError,
    EmbeddedSeamlessData,
    IntersectionDataset,
    UnionDataset,
)


class TestEmbeddedSeamlessData:
    @pytest.fixture
    def dataset(self, test_data: Callable[[str], str]) -> EmbeddedSeamlessData:
        paths = test_data('esd')
        return EmbeddedSeamlessData(paths, transforms=torch.nn.Identity())

    def test_getitem(self, dataset: EmbeddedSeamlessData) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_len(self, dataset: EmbeddedSeamlessData) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: EmbeddedSeamlessData) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: EmbeddedSeamlessData) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: EmbeddedSeamlessData) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EmbeddedSeamlessData(tmp_path)

    def test_invalid_query(self, dataset: EmbeddedSeamlessData) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
