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
    EUDEM,
    DatasetNotFoundError,
    IntersectionDataset,
    UnionDataset,
)


class TestEUDEM:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> EUDEM:
        md5s = {'eu_dem_v11_E30N10.zip': 'ef148466c02197a08be169eaad186591'}
        monkeypatch.setattr(EUDEM, 'md5s', md5s)
        zipfile = os.path.join('tests', 'data', 'eudem', 'eu_dem_v11_E30N10.zip')
        shutil.copy(zipfile, tmp_path)
        root = tmp_path
        transforms = nn.Identity()
        return EUDEM(root, transforms=transforms)

    def test_getitem(self, dataset: EUDEM) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: EUDEM) -> None:
        assert len(dataset) == 1

    def test_extracted_already(self, dataset: EUDEM) -> None:
        assert isinstance(dataset.paths, Path)
        zipfile = os.path.join(dataset.paths, 'eu_dem_v11_E30N10.zip')
        shutil.unpack_archive(zipfile, dataset.paths, 'zip')
        EUDEM(dataset.paths)

    def test_no_dataset(self, tmp_path: Path) -> None:
        shutil.rmtree(tmp_path)
        os.makedirs(tmp_path)
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            EUDEM(tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'eu_dem_v11_E30N10.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            EUDEM(tmp_path, checksum=True)

    def test_and(self, dataset: EUDEM) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: EUDEM) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: EUDEM) -> None:
        query = dataset.bounds
        x = dataset[query]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: EUDEM) -> None:
        query = dataset.bounds
        x = dataset[query]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_query(self, dataset: EUDEM) -> None:
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[100:100, 100:100, pd.Timestamp.min : pd.Timestamp.min]
