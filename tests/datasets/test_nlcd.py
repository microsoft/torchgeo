# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pyproj import CRS
from pytest import MonkeyPatch

from torchgeo.datasets import (
    NLCD,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


class TestNLCD:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> NLCD:
        md5s = {
            2011: '3346297a3cb53c9bd1c7e03b2e6e2d74',
            2019: 'a307cdaa1add9dae05efe02fec4c33bb',
        }
        monkeypatch.setattr(NLCD, 'md5s', md5s)

        url = os.path.join('tests', 'data', 'nlcd', 'Annual_NLCD_LndCov_{}_CU_C1V0.tif')
        monkeypatch.setattr(NLCD, 'url', url)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        root = tmp_path
        transforms = nn.Identity()
        return NLCD(
            root,
            transforms=transforms,
            download=True,
            checksum=True,
            years=[2011, 2019],
        )

    def test_getitem(self, dataset: NLCD) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: NLCD) -> None:
        assert len(dataset) == 2

    def test_classes(self) -> None:
        root = os.path.join('tests', 'data', 'nlcd')
        classes = list(NLCD.cmap.keys())[:5]
        ds = NLCD(root, years=[2019], classes=classes)
        sample = ds[ds.bounds]
        mask = sample['mask']
        assert mask.max() < len(classes)

    def test_and(self, dataset: NLCD) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: NLCD) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_extracted(self, dataset: NLCD) -> None:
        NLCD(dataset.paths, download=True, years=[2019])

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join(
            'tests', 'data', 'nlcd', 'Annual_NLCD_LndCov_2019_CU_C1V0.tif'
        )
        root = tmp_path
        shutil.copy(pathname, root)
        NLCD(root, years=[2019])

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match='NLCD data product only exists for the following years:',
        ):
            NLCD(tmp_path, years=[1984])

    def test_invalid_classes(self) -> None:
        with pytest.raises(AssertionError):
            NLCD(classes=[-1])

        with pytest.raises(AssertionError):
            NLCD(classes=[11])

    def test_plot(self, dataset: NLCD) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: NLCD) -> None:
        query = dataset.bounds
        x = dataset[query]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            NLCD(tmp_path)

    def test_invalid_query(self, dataset: NLCD) -> None:
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
