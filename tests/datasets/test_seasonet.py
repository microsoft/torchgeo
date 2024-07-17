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
from torchgeo.datasets import DatasetNotFoundError, RGBBandsMissingError, SeasoNet


def download_url(url: str, root: str, md5: str, *args: str, **kwargs: str) -> None:
    shutil.copy(url, root)
    torchgeo.datasets.utils.check_integrity(
        os.path.join(root, os.path.basename(url)), md5
    )


class TestSeasoNet:
    @pytest.fixture(
        params=zip(
            ['train', 'val', 'test'],
            [{'Spring'}, {'Summer', 'Fall', 'Winter', 'Snow'}, SeasoNet.all_seasons],
            [SeasoNet.all_bands, ['10m_IR', '10m_RGB', '60m'], ['10m_RGB']],
            [[1], [2], [1, 2]],
            [1, 3, 5],
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SeasoNet:
        monkeypatch.setattr(torchgeo.datasets.seasonet, 'download_url', download_url)
        monkeypatch.setitem(
            SeasoNet.metadata[0], 'md5', '836a0896eba0e3005208f3fd180e429d'
        )
        monkeypatch.setitem(
            SeasoNet.metadata[1], 'md5', '405656c8c19d822620bbb9f92e687337'
        )
        monkeypatch.setitem(
            SeasoNet.metadata[2], 'md5', 'dc0dda18de019a9f50a794b8b4060a3b'
        )
        monkeypatch.setitem(
            SeasoNet.metadata[3], 'md5', 'a70abca62e78eb1591555809dc81d91d'
        )
        monkeypatch.setitem(
            SeasoNet.metadata[4], 'md5', '67651cc9095207e07ea4db1a71f0ebc2'
        )
        monkeypatch.setitem(
            SeasoNet.metadata[5], 'md5', '576324ba1c32a7e9ba858f1c2577cf2a'
        )
        monkeypatch.setitem(
            SeasoNet.metadata[6], 'md5', '48ff6e9e01fdd92379e5712e4f336ea8'
        )
        monkeypatch.setitem(
            SeasoNet.metadata[0],
            'url',
            os.path.join('tests', 'data', 'seasonet', 'spring.zip'),
        )
        monkeypatch.setitem(
            SeasoNet.metadata[1],
            'url',
            os.path.join('tests', 'data', 'seasonet', 'summer.zip'),
        )
        monkeypatch.setitem(
            SeasoNet.metadata[2],
            'url',
            os.path.join('tests', 'data', 'seasonet', 'fall.zip'),
        )
        monkeypatch.setitem(
            SeasoNet.metadata[3],
            'url',
            os.path.join('tests', 'data', 'seasonet', 'winter.zip'),
        )
        monkeypatch.setitem(
            SeasoNet.metadata[4],
            'url',
            os.path.join('tests', 'data', 'seasonet', 'snow.zip'),
        )
        monkeypatch.setitem(
            SeasoNet.metadata[5],
            'url',
            os.path.join('tests', 'data', 'seasonet', 'splits.zip'),
        )
        monkeypatch.setitem(
            SeasoNet.metadata[6],
            'url',
            os.path.join('tests', 'data', 'seasonet', 'meta.csv'),
        )
        root = str(tmp_path)
        split, seasons, bands, grids, concat_seasons = request.param
        transforms = nn.Identity()
        return SeasoNet(
            root=root,
            split=split,
            seasons=seasons,
            bands=bands,
            grids=grids,
            concat_seasons=concat_seasons,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: SeasoNet) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].shape == (dataset.concat_seasons * dataset.channels, 120, 120)
        assert x['mask'].shape == (120, 120)

    def test_len(self, dataset: SeasoNet, request: SubRequest) -> None:
        num_seasons = len(request.node.callspec.params['dataset'][1])
        num_grids = len(request.node.callspec.params['dataset'][3])
        if dataset.concat_seasons == 1:
            assert len(dataset) == num_grids * num_seasons
        else:
            assert len(dataset) == num_grids

    def test_add(self, dataset: SeasoNet, request: SubRequest) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        num_seasons = len(request.node.callspec.params['dataset'][1])
        num_grids = len(request.node.callspec.params['dataset'][3])
        if dataset.concat_seasons == 1:
            assert len(ds) == num_grids * num_seasons * 2
        else:
            assert len(ds) == num_grids * 2

    def test_already_extracted(self, dataset: SeasoNet) -> None:
        SeasoNet(root=dataset.root)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        paths = os.path.join('tests', 'data', 'seasonet', '*.*')
        root = str(tmp_path)
        for path in glob.iglob(paths):
            shutil.copy(path, root)
        SeasoNet(root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SeasoNet(str(tmp_path), download=False)

    def test_out_of_bounds(self, dataset: SeasoNet) -> None:
        with pytest.raises(IndexError):
            dataset[5]

    def test_invalid_seasons(self) -> None:
        with pytest.raises(AssertionError):
            SeasoNet(seasons=('Salt', 'Pepper'))

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            SeasoNet(bands=['30s_TOMARS', '9in_NAILS'])

    def test_invalid_grids(self) -> None:
        with pytest.raises(AssertionError):
            SeasoNet(grids=[42])

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            SeasoNet(split='banana')

    def test_invalid_concat(self) -> None:
        with pytest.raises(AssertionError):
            SeasoNet(seasons={'Spring', 'Winter', 'Snow'}, concat_seasons=4)

    def test_plot(self, dataset: SeasoNet) -> None:
        x = dataset[0]
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        dataset.plot(x, show_legend=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()

    def test_plot_no_rgb(self) -> None:
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            root = os.path.join('tests', 'data', 'seasonet')
            dataset = SeasoNet(root, bands=['10m_IR'])
            x = dataset[0]
            dataset.plot(x)
