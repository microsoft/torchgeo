# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
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
            2011: '531fcba859a0bee6bfeb362a26f6a07f',
            2019: '19a64a25e3c36d8d51b40ab59bddc1ec',
        }
        monkeypatch.setattr(NLCD, 'md5s', md5s)
        url = os.path.join('tests', 'data', 'nlcd', 'Annual_NLCD_LndCov_{}_CU_C1V1.zip')
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

    def test_full_year(self, dataset: NLCD) -> None:
        time = pd.Timestamp(2019, 6, 1)
        query = (dataset.bounds[0], dataset.bounds[1], slice(time, time))
        dataset[query]

    def test_already_extracted(self, dataset: NLCD) -> None:
        NLCD(dataset.paths, years=[2011, 2019])

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'nlcd', '*_CU_C1V1.zip')
        root = tmp_path
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)

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
            IndexError, match=r'query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
