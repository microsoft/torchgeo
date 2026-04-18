# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import DatasetNotFoundError, MapInWild


class TestMapInWild:
    @pytest.fixture(params=['train', 'validation', 'test'])
    def dataset(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, request: SubRequest
    ) -> MapInWild:
        urls = os.path.join('tests', 'data', 'mapinwild')
        monkeypatch.setattr(MapInWild, 'url', urls)

        root = tmp_path
        split = request.param

        transforms = nn.Identity()
        modality = [
            'mask',
            'viirs',
            'esa_wc',
            's2_winter',
            's1',
            's2_summer',
            's2_spring',
            's2_autumn',
            's2_temporal_subset',
        ]
        return MapInWild(
            root, modality=modality, split=split, transforms=transforms, download=True
        )

    def test_getitem(self, dataset: MapInWild) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].ndim == 3

    def test_len(self, dataset: MapInWild) -> None:
        assert len(dataset) == 1

    def test_add(self, dataset: MapInWild) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            MapInWild(root=tmp_path)

    def test_downloaded_not_extracted(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'mapinwild', '*', '*')
        pathname_glob = glob.glob(pathname)
        root = tmp_path
        for zipfile in pathname_glob:
            shutil.copy(zipfile, root)
        MapInWild(root, download=False)

    def test_corrupted(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'mapinwild', '**', '*.zip')
        pathname_glob = glob.glob(pathname, recursive=True)
        root = tmp_path
        for zipfile in pathname_glob:
            shutil.copy(zipfile, root)
        splitfile = os.path.join(
            'tests', 'data', 'mapinwild', 'split_IDs', 'split_IDs.csv'
        )
        shutil.copy(splitfile, root)
        with open(os.path.join(tmp_path, 'mask.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            MapInWild(root=tmp_path, checksum=True)

    def test_already_downloaded(self, dataset: MapInWild, tmp_path: Path) -> None:
        MapInWild(root=tmp_path, modality=dataset.modality, download=True)

    def test_plot(self, dataset: MapInWild) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()
