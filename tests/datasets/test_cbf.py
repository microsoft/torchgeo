# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import (
    CanadianBuildingFootprints,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


class TestCanadianBuildingFootprints:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> CanadianBuildingFootprints:
        monkeypatch.setattr(
            CanadianBuildingFootprints, 'provinces_territories', ['Alberta']
        )
        monkeypatch.setattr(
            CanadianBuildingFootprints, 'md5s', ['25091d1f051baa30d8f2026545cfb696']
        )
        url = os.path.join('tests', 'data', 'cbf') + os.sep
        monkeypatch.setattr(CanadianBuildingFootprints, 'url', url)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        root = tmp_path
        transforms = nn.Identity()
        return CanadianBuildingFootprints(
            root, res=(0.1, 0.1), transforms=transforms, download=True, checksum=True
        )

    def test_getitem(self, dataset: CanadianBuildingFootprints) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: CanadianBuildingFootprints) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: CanadianBuildingFootprints) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: CanadianBuildingFootprints) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: CanadianBuildingFootprints) -> None:
        CanadianBuildingFootprints(dataset.paths, download=True)

    def test_plot(self, dataset: CanadianBuildingFootprints) -> None:
        index = dataset.bounds
        x = dataset[index]
        dataset.plot(x, suptitle='Test')

    def test_plot_prediction(self, dataset: CanadianBuildingFootprints) -> None:
        index = dataset.bounds
        x = dataset[index]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CanadianBuildingFootprints(tmp_path)

    def test_invalid_index(self, dataset: CanadianBuildingFootprints) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[2:2, 2:2, pd.Timestamp.min : pd.Timestamp.min]
