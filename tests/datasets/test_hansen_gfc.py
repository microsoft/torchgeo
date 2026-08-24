# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from torch import nn

from torchgeo.datasets import (
    DatasetNotFoundError,
    HansenGlobalForestChange,
    IntersectionDataset,
    UnionDataset,
)


class TestHansenGlobalForestChange:
    @pytest.fixture
    def dataset(self, tmp_path: Path) -> HansenGlobalForestChange:
        src = os.path.join('tests', 'data', 'hansen_gfc')
        for name in os.listdir(src):
            if name.endswith('.tif'):
                shutil.copy(os.path.join(src, name), tmp_path)

        transforms = nn.Identity()
        return HansenGlobalForestChange(tmp_path, transforms=transforms)

    def test_getitem(self, dataset: HansenGlobalForestChange) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: HansenGlobalForestChange) -> None:
        assert len(dataset) >= 1

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            HansenGlobalForestChange(tmp_path)

    def test_and(self, dataset: HansenGlobalForestChange) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: HansenGlobalForestChange) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: HansenGlobalForestChange) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_no_titles(self, dataset: HansenGlobalForestChange) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, show_titles=False)
        plt.close()

    def test_plot_prediction(self, dataset: HansenGlobalForestChange) -> None:
        x = dataset[dataset.bounds]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_index(self, dataset: HansenGlobalForestChange) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[100:100, 100:100, pd.Timestamp.min : pd.Timestamp.min]
