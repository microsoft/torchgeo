# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    SouthAmericaSoybean,
    UnionDataset,
)


class TestSouthAmericaSoybean:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, test_data: Callable[[str], str]
    ) -> SouthAmericaSoybean:
        transforms = nn.Identity()
        url = os.path.join(
            test_data('south_america_soybean'), 'SouthAmerica_Soybean_{}.tif'
        )

        monkeypatch.setattr(SouthAmericaSoybean, 'url', url)
        root = tmp_path
        return SouthAmericaSoybean(
            paths=root, years=[2002, 2021], transforms=transforms, download=True
        )

    def test_getitem(self, dataset: SouthAmericaSoybean) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: SouthAmericaSoybean) -> None:
        assert len(dataset) == 2

    def test_and(self, dataset: SouthAmericaSoybean) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: SouthAmericaSoybean) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: SouthAmericaSoybean) -> None:
        SouthAmericaSoybean(dataset.paths, download=True)

    def test_already_downloaded(
        self, tmp_path: Path, test_data: Callable[[str], str]
    ) -> None:
        pathname = os.path.join(
            test_data('south_america_soybean'), 'SouthAmerica_Soybean_2002.tif'
        )
        root = tmp_path
        shutil.copy(pathname, root)
        SouthAmericaSoybean(root)

    def test_plot(self, dataset: SouthAmericaSoybean) -> None:
        index = dataset.bounds
        x = dataset[index]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: SouthAmericaSoybean) -> None:
        index = dataset.bounds
        x = dataset[index]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SouthAmericaSoybean(tmp_path)

    def test_invalid_index(self, dataset: SouthAmericaSoybean) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
