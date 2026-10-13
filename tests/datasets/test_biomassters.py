# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import BioMassters, BioMassters100, DatasetNotFoundError


class TestBioMassters:
    @pytest.fixture(
        params=product(['train', 'test'], [['S1'], ['S2'], ['S1', 'S2']], [True, False])
    )
    def dataset(self, request: SubRequest) -> BioMassters:
        root = os.path.join('tests', 'data', 'biomassters')
        split, sensors, as_time_series = request.param
        return BioMassters(
            root, split=split, sensors=sensors, as_time_series=as_time_series
        )

    def test_len_of_ds(self, dataset: BioMassters) -> None:
        assert len(dataset) > 0

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BioMassters(tmp_path)

    def test_plot(self, dataset: BioMassters) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        if dataset.split == 'train':
            sample['prediction'] = sample['label']
        dataset.plot(sample)
        plt.close()
        dataset.plot(sample, show_titles=False)
        plt.close()


class TestBioMassters100:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> BioMassters100:
        source = Path('tests/data/biomassters')
        archive_root = tmp_path / 'archive'
        directory = archive_root / BioMassters100.directory
        directory.mkdir(parents=True)
        shutil.copy(source / BioMassters.metadata_filename, directory)
        for name in ['train_features', 'test_features', 'train_agbm']:
            shutil.copytree(source / name, directory / name)
        archive = shutil.make_archive(
            str(tmp_path / BioMassters100.directory),
            'zip',
            root_dir=archive_root,
            base_dir=BioMassters100.directory,
        )
        monkeypatch.setattr(BioMassters100, 'url', archive)
        return BioMassters100(tmp_path / 'download', download=True, checksum=False)

    def test_getitem(self, dataset: BioMassters100) -> None:
        assert len(dataset) > 0
        assert dataset[0]

    def test_already_extracted(self, dataset: BioMassters100) -> None:
        root = Path(dataset.root).parent
        assert BioMassters100(root, checksum=False)

    def test_extract_archive(self, dataset: BioMassters100) -> None:
        root = Path(dataset.root).parent
        shutil.rmtree(dataset.root)
        assert BioMassters100(root, checksum=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BioMassters100(tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        (tmp_path / BioMassters100.filename).write_text('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            BioMassters100(tmp_path)
