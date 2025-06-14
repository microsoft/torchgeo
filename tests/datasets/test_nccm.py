# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pyproj import CRS
from pytest import MonkeyPatch

from torchgeo.datasets import (
    NCCM,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


class TestNCCM:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> NCCM:
        md5s = {
            2017: 'ae5c390d0ffb8970d544b8a09142759f',
            2018: '0d453bdb8ea5b7318c33e62513760580',
            2019: 'd4ab7ab00bb57623eafb6b27747e5639',
        }
        monkeypatch.setattr(NCCM, 'md5s', md5s)
        urls = {
            2017: os.path.join('tests', 'data', 'nccm', 'CDL2017_clip.tif'),
            2018: os.path.join('tests', 'data', 'nccm', 'CDL2018_clip1.tif'),
            2019: os.path.join('tests', 'data', 'nccm', 'CDL2019_clip.tif'),
        }
        monkeypatch.setattr(NCCM, 'urls', urls)
        transforms = nn.Identity()
        root = tmp_path
        return NCCM(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: NCCM) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: NCCM) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: NCCM) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: NCCM) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: NCCM) -> None:
        NCCM(dataset.paths, download=True)

    def test_already_downloaded(self, dataset: NCCM) -> None:
        NCCM(dataset.paths, download=True)

    def test_plot(self, dataset: NCCM) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: NCCM) -> None:
        query = dataset.bounds
        x = dataset[query]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            NCCM(tmp_path)

    def test_invalid_query(self, dataset: NCCM) -> None:
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
