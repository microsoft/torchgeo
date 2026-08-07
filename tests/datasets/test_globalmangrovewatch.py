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
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import (
    DatasetNotFoundError,
    GlobalMangroveWatch,
    IntersectionDataset,
    UnionDataset,
)


class TestGlobalMangroveWatch:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> GlobalMangroveWatch:
        url = os.path.join(
            'tests', 'data', 'globalmangrovewatch', 'gmw_v3_{}_gtiff.zip'
        )
        monkeypatch.setattr(GlobalMangroveWatch, 'url', url)
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
        ds = GlobalMangroveWatch(dataset.paths, years=[1996])
        assert len(ds) == 1

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join(
            'tests', 'data', 'globalmangrovewatch', 'gmw_v3_*_gtiff.zip'
        )
        for zipfile in glob.iglob(pathname):
            shutil.copy(zipfile, tmp_path)
        GlobalMangroveWatch(tmp_path, years=[2020])

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'gmw_v3_2020_gtiff.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            GlobalMangroveWatch(tmp_path, years=[2020], checksum=True)

    def test_multiple_paths_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(AssertionError, match='paths must be a single root'):
            GlobalMangroveWatch([tmp_path], years=[2020], download=True)

    def test_invalid_year(self, tmp_path: Path) -> None:
        with pytest.raises(
            AssertionError,
            match='GMW data product only exists for the following years:',
        ):
            GlobalMangroveWatch(tmp_path, years=[2021])

    def test_plot(self, dataset: GlobalMangroveWatch) -> None:
        ds = GlobalMangroveWatch(dataset.paths, years=[1996, 2020], time_series=True)
        x = ds[ds.bounds]
        ds.plot(x, suptitle='Test')
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
