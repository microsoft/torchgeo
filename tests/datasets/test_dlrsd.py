# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import DLRSD, DatasetNotFoundError, DLRSDMultilabel


class TestDLRSD:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> DLRSD:
        url = os.path.join('tests', 'data', 'dlrsd') + os.sep
        monkeypatch.setattr(DLRSD, 'url', url)
        transforms = nn.Identity()
        return DLRSD(tmp_path, transforms, download=True)

    def test_getitem(self, dataset: DLRSD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: DLRSD) -> None:
        assert len(dataset) == 4

    def test_add(self, dataset: DLRSD) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 8

    def test_already_downloaded(self, dataset: DLRSD, tmp_path: Path) -> None:
        DLRSD(tmp_path)

    def test_already_downloaded_not_extracted(
        self, dataset: DLRSD, tmp_path: Path
    ) -> None:
        shutil.rmtree(os.path.join(tmp_path, dataset.directory))
        DLRSD(tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'DLRSD.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            DLRSD(root=tmp_path, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DLRSD(tmp_path)

    def test_plot(self, dataset: DLRSD) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()
        # Cover the float-image-normalized-to-[0, 1] overlay rescaling branch
        x['image'] = x['image'] / 255.0
        dataset.plot(x)
        plt.close()


class TestDLRSDMultilabel:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> DLRSDMultilabel:
        url = os.path.join('tests', 'data', 'dlrsd') + os.sep
        monkeypatch.setattr(DLRSDMultilabel, 'url', url)
        transforms = nn.Identity()
        return DLRSDMultilabel(tmp_path, transforms, download=True)

    def test_getitem(self, dataset: DLRSDMultilabel) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['label'].shape == (17,)

    def test_len(self, dataset: DLRSDMultilabel) -> None:
        assert len(dataset) == 4

    def test_add(self, dataset: DLRSDMultilabel) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 8

    def test_already_downloaded(self, dataset: DLRSDMultilabel, tmp_path: Path) -> None:
        DLRSDMultilabel(tmp_path)

    def test_already_downloaded_not_extracted(
        self, dataset: DLRSDMultilabel, tmp_path: Path
    ) -> None:
        shutil.rmtree(os.path.join(tmp_path, dataset.directory))
        DLRSDMultilabel(tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'DLRSD.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            DLRSDMultilabel(root=tmp_path, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DLRSDMultilabel(tmp_path)

    def test_plot(self, dataset: DLRSDMultilabel) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()
