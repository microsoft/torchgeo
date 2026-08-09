# Copyright (c) TorchGeo Contributors. All rights reserved.

# Licensed under the MIT License.

import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest

from torchgeo.datasets import BioMassters, DatasetNotFoundError


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

    def test_download_constructor(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def mock_download(self: BioMassters) -> None:
            calls.append('download')

        def mock_extract(self: BioMassters) -> None:
            calls.append('extract')

        metadata = pd.DataFrame(
            {
                'satellite': ['S1', 'S2'],
                'split': ['test', 'test'],
                'filename': ['S1_x_0.tif', 'S2_x_0.tif'],
                'chip_id': [1, 1],
                'month': [1, 1],
            }
        )

        monkeypatch.setattr(BioMassters, '_download', mock_download)
        monkeypatch.setattr(BioMassters, '_extract', mock_extract)
        monkeypatch.setattr(
            'torchgeo.datasets.biomassters.pd.read_csv',
            lambda *args, **kwargs: metadata.copy(),
        )

        dataset = BioMassters(tmp_path, split='test', download=True)

        assert dataset.download is True
        assert dataset.checksum is True
        assert calls == ['download', 'extract']

    def test_verify_existing_archives(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for filename in BioMassters.feature_archive_filenames['test']:
            (tmp_path / filename).touch()

        (tmp_path / BioMassters.metadata_filename).touch()

        calls = []

        def mock_extract(self: BioMassters) -> None:
            calls.append('extract')

        monkeypatch.setattr(BioMassters, '_extract', mock_extract)

        dataset = BioMassters.__new__(BioMassters)
        dataset.root = tmp_path
        dataset.split = 'test'

        dataset._verify()

        assert calls == ['extract']

    def test_download_checksum(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def mock_download(
            url: str, root: Path, filename: str | None = None, **kwargs: object
        ) -> None:
            calls.append((filename, kwargs))

        monkeypatch.setattr('torchgeo.datasets.biomassters.download_url', mock_download)

        dataset = BioMassters.__new__(BioMassters)
        dataset.root = tmp_path
        dataset.split = 'test'
        dataset.checksum = True
        dataset._download()

        assert len(calls) == 3

        for filename, kwargs in calls:
            assert kwargs['sha256'] == dataset.checksums[filename]

    def test_download_checksum_train(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def mock_download(
            url: str, root: Path, filename: str | None = None, **kwargs: object
        ) -> None:
            calls.append((filename, kwargs))

        monkeypatch.setattr('torchgeo.datasets.biomassters.download_url', mock_download)

        dataset = BioMassters.__new__(BioMassters)
        dataset.root = tmp_path
        dataset.split = 'train'
        dataset.checksum = True
        dataset._download()

        assert len(calls) == 6

        for filename, kwargs in calls:
            assert kwargs['sha256'] == dataset.checksums[filename]

    def test_download_without_checksum(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = []

        def mock_download(
            url: str, root: Path, filename: str | None = None, **kwargs: object
        ) -> None:
            calls.append((filename, kwargs))

        monkeypatch.setattr('torchgeo.datasets.biomassters.download_url', mock_download)

        dataset = BioMassters.__new__(BioMassters)
        dataset.root = tmp_path
        dataset.split = 'test'
        dataset.checksum = False
        dataset._download()

        assert len(calls) == 3

        for _, kwargs in calls:
            assert kwargs['sha256'] is None

    def test_extract_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feature_filenames = BioMassters.feature_archive_filenames['test']

        for filename in feature_filenames:
            (tmp_path / filename).write_bytes(b'test')

        calls = []

        def mock_extract(path: str, root: Path) -> None:
            calls.append((path, root))

        monkeypatch.setattr(
            'torchgeo.datasets.biomassters.extract_archive', mock_extract
        )

        dataset = BioMassters.__new__(BioMassters)
        dataset.root = tmp_path
        dataset.split = 'test'
        dataset._extract()

        combined_path = os.path.join(tmp_path, 'test_features.tar.gz')

        assert calls == [(combined_path, tmp_path)]
        assert not os.path.exists(combined_path)

    def test_extract_train(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        feature_filenames = BioMassters.feature_archive_filenames['train']

        for filename in feature_filenames:
            (tmp_path / filename).write_bytes(b'train')

        target_filename = BioMassters.target_archive_filenames['train']
        (tmp_path / target_filename).touch()

        calls = []

        def mock_extract(path: str, root: Path) -> None:
            calls.append((path, root))

        monkeypatch.setattr(
            'torchgeo.datasets.biomassters.extract_archive', mock_extract
        )

        dataset = BioMassters.__new__(BioMassters)
        dataset.root = tmp_path
        dataset.split = 'train'
        dataset._extract()

        combined_path = os.path.join(tmp_path, 'train_features.tar.gz')
        target_path = os.path.join(tmp_path, target_filename)

        assert calls == [(combined_path, tmp_path), (target_path, tmp_path)]
        assert not os.path.exists(combined_path)

    def test_plot(self, dataset: BioMassters) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        if dataset.split == 'train':
            sample['prediction'] = sample['label']
        dataset.plot(sample)
        plt.close()
