# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
import torch
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    IOBench,
    RGBBandsMissingError,
    UnionDataset,
)


class TestIOBench:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> IOBench:
        url = os.path.join('tests', 'data', 'iobench', '{}.tar.gz')
        monkeypatch.setattr(IOBench, 'url', url)
        root = tmp_path
        transforms = nn.Identity()
        return IOBench(root, transforms=transforms, download=True)

    def test_getitem(self, dataset: IOBench) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: IOBench) -> None:
        assert len(dataset) == 1

    def test_and(self, dataset: IOBench) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: IOBench) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: IOBench) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_already_extracted(self, dataset: IOBench) -> None:
        IOBench(dataset.root, download=True)

    def test_already_downloaded(self, tmp_path: Path) -> None:
        pathname = os.path.join('tests', 'data', 'iobench', '*.tar.gz')
        root = tmp_path
        for tarfile in glob.iglob(pathname):
            shutil.copy(tarfile, root)
        IOBench(root)

    def test_download_checksum(
        self, dataset: IOBench, monkeypatch: MonkeyPatch
    ) -> None:
        checksum = ''

        def download_url(url: str, root: Path, sha256: str | None = None) -> None:
            nonlocal checksum
            checksum = sha256 or ''

        monkeypatch.setattr('torchgeo.datasets.iobench.download_url', download_url)
        dataset.checksum = True
        dataset._download()
        assert checksum == dataset.sha256s[dataset.split]

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            IOBench(tmp_path)

    def test_invalid_index(self, dataset: IOBench) -> None:
        with pytest.raises(
            IndexError, match=r'index: .* not found in dataset with bounds:'
        ):
            dataset[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_rgb_bands_absent_plot(self, dataset: IOBench) -> None:
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            print(dataset.root)
            ds = IOBench(dataset.root, bands=['SR_B1', 'SR_B2', 'SR_B3'])
            x = ds[ds.bounds]
            ds.plot(x, suptitle='Test')
            plt.close()
