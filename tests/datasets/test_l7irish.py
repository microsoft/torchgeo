# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
import torch.nn as nn
from pyproj import CRS
from pytest import MonkeyPatch

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    L7Irish,
    RGBBandsMissingError,
    UnionDataset,
)


class TestL7Irish:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> L7Irish:
        md5s = {
            'austral': '0485d6045f6b508068ef8daf9e5a5326',
            'boreal': '5798f32545d7166564c4c4429357b840',
        }

        url = os.path.join('tests', 'data', 'l7irish', '{}.tar.gz')
        monkeypatch.setattr(L7Irish, 'url', url)
        monkeypatch.setattr(L7Irish, 'md5s', md5s)
        root = tmp_path
        transforms = nn.Identity()
        return L7Irish(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: L7Irish) -> None:
        assert len(dataset) == 5

    def test_and(self, dataset: L7Irish) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: L7Irish) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_already_extracted(self, dataset: L7Irish) -> None:
        paths = cast(str, dataset.paths)
        L7Irish(paths, download=True)
        L7Irish([paths], download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'l7irish', '*.tar.gz')
        root = tmp_path
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        L7Irish(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            L7Irish(tmp_path)

    def test_plot_prediction(self, dataset: L7Irish) -> None:
        x = dataset[dataset.bounds]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_query(self, dataset: L7Irish) -> None:
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_rgb_bands_absent_plot(self, dataset: L7Irish) -> None:
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            ds = L7Irish(dataset.paths, bands=['B10', 'B20', 'B50'])
            x = ds[ds.bounds]
            ds.plot(x, suptitle='Test')
            plt.close()
