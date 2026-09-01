# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from torch import nn

from tests.data.tessera.data import ensure_tessera_data
from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    TesseraEmbeddings,
    UnionDataset,
)


class TestTesseraEmbeddings:
    @pytest.fixture
    def dataset(self) -> TesseraEmbeddings:
        paths = os.path.join('tests', 'data', 'tessera')
        transforms = nn.Identity()
        return TesseraEmbeddings(paths, transforms=transforms)

    def test_len(self, dataset: TesseraEmbeddings) -> None:
        assert len(dataset) == 1

    def test_getitem(self, dataset: TesseraEmbeddings) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)

    def test_and(self, dataset: TesseraEmbeddings) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: TesseraEmbeddings) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: TesseraEmbeddings) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            TesseraEmbeddings(tmp_path)

    def test_invalid_index(self, dataset: TesseraEmbeddings) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_fixture_uses_fixed_structure(self, tmp_path: Path) -> None:
        ensure_tessera_data(tmp_path)

        filename = (
            tmp_path
            / 'global_0.1_degree_representation'
            / '2024'
            / 'grid_0.05_51.35'
            / 'grid_0.05_51.35_2024.tiff'
        )
        assert filename.exists()
