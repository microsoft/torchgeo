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
    CDL,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


class TestCDL:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> CDL:
        url = os.path.join('tests', 'data', 'cdl', '{}_30m_cdls.zip')
        monkeypatch.setattr(CDL, 'url', url)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        root = tmp_path
        transforms = nn.Identity()
        return CDL(root, transforms=transforms, download=True, years=[2023, 2022])

    def test_getitem(self, dataset: CDL) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: CDL) -> None:
        assert len(dataset) == 2

    def test_classes(self) -> None:
        root = os.path.join('tests', 'data', 'cdl')
        classes = list(CDL.cmap.keys())[:5]
        ds = CDL(root, years=[2023], classes=classes)
        sample = ds[ds.bounds]
        mask = sample['mask']
        assert mask.max() < len(classes)

    def test_and(self, dataset: CDL) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: CDL) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_full_year(self, dataset: CDL) -> None:
        time = pd.Timestamp(2023, 6, 1)
        index = (dataset.bounds[0], dataset.bounds[1], slice(time, time))
        dataset[index]

    def test_already_extracted(self, dataset: CDL) -> None:
        CDL(dataset.paths, years=[2023, 2022])

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'cdl', '*_30m_cdls.zip')
        root = tmp_path
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, root)
        CDL(root, years=[2023, 2022])

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match='CDL data product only exists for the following years:',
        ):
            CDL(tmp_path, years=[1996])

    def test_invalid_classes(self) -> None:
        with pytest.raises(AssertionError):
            CDL(classes=[-1])

        with pytest.raises(AssertionError):
            CDL(classes=[11])

    def test_plot(self, dataset: CDL) -> None:
        index = dataset.bounds
        x = dataset[index]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: CDL) -> None:
        index = dataset.bounds
        x = dataset[index]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CDL(tmp_path)

    def test_invalid_index(self, dataset: CDL) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]
