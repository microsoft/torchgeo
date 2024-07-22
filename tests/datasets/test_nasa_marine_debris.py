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
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, NASAMarineDebris


class Collection:
    def download(self, output_dir: str, **kwargs: str) -> None:
        glob_path = os.path.join('tests', 'data', 'nasa_marine_debris', '*.tar.gz')
        for tarball in glob.iglob(glob_path):
            shutil.copy(tarball, output_dir)


def fetch(collection_id: str, **kwargs: str) -> Collection:
    return Collection()


class Collection_corrupted:
    def download(self, output_dir: str, **kwargs: str) -> None:
        filenames = NASAMarineDebris.filenames
        for filename in filenames:
            with open(os.path.join(output_dir, filename), 'w') as f:
                f.write('bad')


def fetch_corrupted(collection_id: str, **kwargs: str) -> Collection_corrupted:
    return Collection_corrupted()


class TestNASAMarineDebris:
    @pytest.fixture()
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> NASAMarineDebris:
        radiant_mlhub = pytest.importorskip('radiant_mlhub', minversion='0.3')
        monkeypatch.setattr(radiant_mlhub.Collection, 'fetch', fetch)
        md5s = ['6f4f0d2313323950e45bf3fc0c09b5de', '540cf1cf4fd2c13b609d0355abe955d7']
        monkeypatch.setattr(NASAMarineDebris, 'md5s', md5s)
        root = tmp_path
        transforms = nn.Identity()
        return NASAMarineDebris(root, transforms, download=True, checksum=True)

    def test_getitem(self, dataset: NASAMarineDebris) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['boxes'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert x['boxes'].shape[-1] == 4

    def test_len(self, dataset: NASAMarineDebris) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(
        self, dataset: NASAMarineDebris, tmp_path: Path
    ) -> None:
        NASAMarineDebris(root=tmp_path, download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: NASAMarineDebris, tmp_path: Path
    ) -> None:
        shutil.rmtree(dataset.root)
        os.makedirs(tmp_path, exist_ok=True)
        Collection().download(output_dir=str(tmp_path))
        NASAMarineDebris(root=tmp_path, download=False)

    def test_corrupted_previously_downloaded(self, tmp_path: Path) -> None:
        filenames = NASAMarineDebris.filenames
        for filename in filenames:
            with open(os.path.join(tmp_path, filename), 'w') as f:
                f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset checksum mismatch.'):
            NASAMarineDebris(root=tmp_path, download=False, checksum=True)

    def test_corrupted_new_download(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        with pytest.raises(RuntimeError, match='Dataset checksum mismatch.'):
            radiant_mlhub = pytest.importorskip('radiant_mlhub', minversion='0.3')
            monkeypatch.setattr(radiant_mlhub.Collection, 'fetch', fetch_corrupted)
            NASAMarineDebris(root=tmp_path, download=True, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            NASAMarineDebris(tmp_path)

    def test_plot(self, dataset: NASAMarineDebris) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction_boxes'] = x['boxes'].clone()
        dataset.plot(x)
        plt.close()
