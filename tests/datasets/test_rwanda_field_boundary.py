# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    DatasetNotFoundError,
    RGBBandsMissingError,
    RwandaFieldBoundary,
)
from torchgeo.datasets.utils import Executable


class TestRwandaFieldBoundary:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self,
        azcopy: Executable,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
        request: SubRequest,
    ) -> RwandaFieldBoundary:
        url = os.path.join('tests', 'data', 'rwanda_field_boundary')
        monkeypatch.setattr(RwandaFieldBoundary, 'url', url)
        monkeypatch.setattr(RwandaFieldBoundary, 'splits', {'train': 1, 'test': 1})

        root = str(tmp_path)
        split = request.param
        transforms = nn.Identity()
        return RwandaFieldBoundary(root, split, transforms=transforms, download=True)

    def test_getitem(self, dataset: RwandaFieldBoundary) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        if dataset.split == 'train':
            assert isinstance(x['mask'], torch.Tensor)
        else:
            assert 'mask' not in x

    def test_len(self, dataset: RwandaFieldBoundary) -> None:
        assert len(dataset) == 1

    def test_add(self, dataset: RwandaFieldBoundary) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2

    def test_already_downloaded(self, dataset: RwandaFieldBoundary) -> None:
        RwandaFieldBoundary(root=dataset.root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            RwandaFieldBoundary(str(tmp_path))

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            RwandaFieldBoundary(bands=('foo', 'bar'))

    def test_plot(self, dataset: RwandaFieldBoundary) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()

        if dataset.split == 'train':
            x['prediction'] = x['mask'].clone()
            dataset.plot(x)
            plt.close()

    def test_failed_plot(self, dataset: RwandaFieldBoundary) -> None:
        single_band_dataset = RwandaFieldBoundary(root=dataset.root, bands=('B01',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            x = single_band_dataset[0].copy()
            single_band_dataset.plot(x, suptitle='Test')
