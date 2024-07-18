# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    BeninSmallHolderCashews,
    DatasetNotFoundError,
    RGBBandsMissingError,
)
from torchgeo.datasets.utils import Executable


class TestBeninSmallHolderCashews:
    @pytest.fixture
    def dataset(
        self, azcopy: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> BeninSmallHolderCashews:
        url = os.path.join('tests', 'data', 'technoserve-cashew-benin')
        monkeypatch.setattr(BeninSmallHolderCashews, 'url', url)
        monkeypatch.setattr(BeninSmallHolderCashews, 'dates', ('20191105',))
        monkeypatch.setattr(BeninSmallHolderCashews, 'tile_height', 2)
        monkeypatch.setattr(BeninSmallHolderCashews, 'tile_width', 2)
        root = tmp_path
        transforms = nn.Identity()
        return BeninSmallHolderCashews(root, transforms=transforms, download=True)

    def test_getitem(self, dataset: BeninSmallHolderCashews) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert isinstance(x['x'], torch.Tensor)
        assert isinstance(x['y'], torch.Tensor)

    def test_len(self, dataset: BeninSmallHolderCashews) -> None:
        assert len(dataset) == 1

    def test_add(self, dataset: BeninSmallHolderCashews) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2

    def test_already_downloaded(self, dataset: BeninSmallHolderCashews) -> None:
        BeninSmallHolderCashews(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BeninSmallHolderCashews(tmp_path)

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            BeninSmallHolderCashews(bands=('foo', 'bar'))

    def test_plot(self, dataset: BeninSmallHolderCashews) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()

    def test_failed_plot(self, dataset: BeninSmallHolderCashews) -> None:
        single_band_dataset = BeninSmallHolderCashews(root=dataset.root, bands=('B01',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            x = single_band_dataset[0].copy()
            single_band_dataset.plot(x, suptitle='Test')
