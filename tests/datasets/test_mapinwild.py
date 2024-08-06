# Copyright (c) Microsoft Corporation. All rights reserved.
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

import torchgeo.datasets.utils
from torchgeo.datasets import DatasetNotFoundError, MapInWild


def download_url(url: str, root: str | Path, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)


class TestMapInWild:
    @pytest.fixture(params=['train', 'validation', 'test'])
    def dataset(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, request: SubRequest
    ) -> MapInWild:
        monkeypatch.setattr(torchgeo.datasets.mapinwild, 'download_url', download_url)

        md5s = {
            'ESA_WC.zip': '3a1e696353d238c50996958855da02fc',
            'VIIRS.zip': 'e8b0e230edb1183c02092357af83bd52',
            'mask.zip': '15245bb6368d27dbb4bd16310f4604fa',
            's1_part1.zip': 'e660da4175518af993b63644e44a9d03',
            's1_part2.zip': '620cf0a7d598a2893bc7642ad7ee6087',
            's2_autumn_part1.zip': '624b6cf0191c5e0bc0d51f92b568e676',
            's2_autumn_part2.zip': 'f848c62b8de36f06f12fb6b1b065c7b6',
            's2_spring_part1.zip': '3296f3a7da7e485708dd16be91deb111',
            's2_spring_part2.zip': 'd27e94387a59f0558fe142a791682861',
            's2_summer_part1.zip': '41d783706c3c1e4238556a772d3232fb',
            's2_summer_part2.zip': '3495c87b67a771cfac5153d1958daa0c',
            's2_temporal_subset_part1.zip': '06fa463888cb033011a06cf69f82273e',
            's2_temporal_subset_part2.zip': '93e5383adeeea27f00051ecf110fcef8',
            's2_winter_part1.zip': '617abe1c6ad8d38725aa27c9dcc38ceb',
            's2_winter_part2.zip': '4e40d7bb0eec4ddea0b7b00314239a49',
            'split_IDs.csv': 'ca22c3d30d0b62e001ed0c327c147127',
        }

        monkeypatch.setattr(MapInWild, 'md5s', md5s)

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
            root,
            modality=modality,
            split=split,
            transforms=transforms,
            download=True,
            checksum=True,
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

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            MapInWild(split='foo')

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
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            MapInWild(root=tmp_path, download=True, checksum=True)

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
