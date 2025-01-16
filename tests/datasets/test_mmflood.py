# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from rasterio.crs import CRS

from torchgeo.datasets import (
    BoundingBox,
    DatasetNotFoundError,
    IntersectionDataset,
    MMFlood,
    UnionDataset,
)


class TestMMFlood:
    @pytest.fixture(
        params=product([True, False], [True, False], ['train', 'val', 'test'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> MMFlood:
        dataset_root = os.path.join('tests', 'data', 'mmflood/')
        url = os.path.join(dataset_root)

        monkeypatch.setattr(MMFlood, 'url', url)
        monkeypatch.setattr(MMFlood, '_nparts', 2)

        include_dem, include_hydro, split = request.param
        root = tmp_path
        return MMFlood(
            root,
            split=split,
            include_dem=include_dem,
            include_hydro=include_hydro,
            transforms=nn.Identity(),
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: MMFlood) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        nchannels = 2

        # If DEM is included and hydro is included, check if 4 channels are present,
        # If only one between DEM or hydro is included, check if 3 channels are present
        # 2 otherwise
        if dataset.include_dem:
            nchannels += 1
        if dataset.include_hydro:
            nchannels += 1
        assert x['image'].size(0) == nchannels

    def test_len(self, dataset: MMFlood) -> None:
        if dataset.split == 'train':
            if not dataset.include_hydro:
                assert len(dataset) == 5
            else:
                assert len(dataset) == 2
        elif dataset.split == 'val':
            if not dataset.include_hydro:
                assert len(dataset) == 2
            else:
                assert len(dataset) == 1
        else:
            assert len(dataset) == 1

    def test_and(self, dataset: MMFlood) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: MMFlood) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_already_downloaded(self, dataset: MMFlood) -> None:
        MMFlood(root=dataset.root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            MMFlood(tmp_path)

    def test_plot(self, dataset: MMFlood) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: MMFlood) -> None:
        x = dataset[dataset.bounds]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()

    def test_invalid_query(self, dataset: MMFlood) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]
