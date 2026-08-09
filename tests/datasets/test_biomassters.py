# Copyright (c) TorchGeo Contributors. All rights reserved.

# Licensed under the MIT License.

import os
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
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

    def test_plot(self, dataset: BioMassters) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        if dataset.split == 'train':
            sample['prediction'] = sample['label']
        dataset.plot(sample)
        plt.close()
