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
    DatasetNotFoundError,
    GlobalMangroveWatch,
    IntersectionDataset,
    UnionDataset,
)


class TestGlobalMangroveWatch:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> GlobalMangroveWatch:
        url = os.path.join(
            'tests', 'data', 'globalmangrovewatch', 'gmw_v3_{}_gtiff.zip'
        )
        monkeypatch.setattr(GlobalMangroveWatch, 'url', url)
        monkeypatch.setattr(plt, 'show', lambda *args: None)
        transforms = nn.Identity()
        return GlobalMangroveWatch(
            tmp_path, transforms=transforms, download=True, years=[1996, 2020]
        )

    def test_getitem(self, dataset: GlobalMangroveWatch) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: GlobalMangroveWatch) -> None:
        assert len(dataset) == 2

    def test_and(self, dataset: GlobalMangroveWatch) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: GlobalMangroveWatch) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_full_year(self, dataset: GlobalMangroveWatch) -> None:
        time = pd.Timestamp(2020, 6, 1)
        index = (dataset.bounds[0], dataset.bounds[1], slice(time, time))
        dataset[index]

    def test_already_extracted(self, dataset: GlobalMangroveWatch) -> None:
        GlobalMangroveWatch(dataset.paths, years=[1996, 2020])

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join(
            'tests', 'data', 'globalmangrovewatch', 'gmw_v3_*_gtiff.zip'
        )
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, tmp_path)
        GlobalMangroveWatch(tmp_path, years=[2020])

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match='GMW data product only exists for the following years:',
        ):
            GlobalMangroveWatch(tmp_path, years=[2021])

    def test_plot(self, dataset: GlobalMangroveWatch) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: GlobalMangroveWatch) -> None:
        x = dataset[dataset.bounds]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            GlobalMangroveWatch(tmp_path)

    def test_invalid_index(self, dataset: GlobalMangroveWatch) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]