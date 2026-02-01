# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from matplotlib import pyplot as plt
from pytest import MonkeyPatch
from torch.utils.data import ConcatDataset

from torchgeo.datasets import OSCD, OSCD100, DatasetNotFoundError, RGBBandsMissingError


class TestOSCD:
    @pytest.fixture(
        params=list(
            product(
                [
                    (OSCD, OSCD.all_bands),
                    (OSCD, OSCD.rgb_bands),
                    (OSCD100, OSCD100.all_bands),
                ],
                ['train', 'test'],
            )
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> OSCD:
        (cls, bands), split = request.param

        urls = {
            'Onera Satellite Change Detection dataset - Images.zip': os.path.join(
                'tests',
                'data',
                'oscd',
                'Onera Satellite Change Detection dataset - Images.zip',
            ),
            'Onera Satellite Change Detection dataset - Train Labels.zip': os.path.join(
                'tests',
                'data',
                'oscd',
                'Onera Satellite Change Detection dataset - Train Labels.zip',
            ),
            'Onera Satellite Change Detection dataset - Test Labels.zip': os.path.join(
                'tests',
                'data',
                'oscd',
                'Onera Satellite Change Detection dataset - Test Labels.zip',
            ),
        }
        monkeypatch.setattr(cls, 'urls', urls)

        return cls(
            root=tmp_path,
            split=split,
            bands=bands,
            transforms=nn.Identity(),
            download=True,
        )

    def test_getitem(self, dataset: OSCD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].ndim == 4
        assert isinstance(x['mask'], torch.Tensor)
        assert x['mask'].ndim == 3

        if dataset.bands == OSCD.rgb_bands:
            assert x['image'].shape[1] == 3
        else:
            assert x['image'].shape[1] == 13

    def test_len(self, dataset: OSCD) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 4
        else:
            assert len(dataset) == 2

    def test_add(self, dataset: OSCD) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)

    def test_already_extracted(self, dataset: OSCD) -> None:
        type(dataset)(root=dataset.root, download=True)

    def test_already_downloaded(self, dataset: OSCD) -> None:
        shutil.rmtree(dataset.root)
        dataset.root.mkdir()
        for zipfile in [
            'Onera Satellite Change Detection dataset - Images.zip',
            'Onera Satellite Change Detection dataset - Train Labels.zip',
            'Onera Satellite Change Detection dataset - Test Labels.zip',
        ]:
            shutil.copy(os.path.join('tests', 'data', 'oscd', zipfile), dataset.root)
        type(dataset)(root=dataset.root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OSCD(tmp_path)

    def test_not_downloaded_oscd100(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OSCD100(tmp_path)

    def test_plot(self, dataset: OSCD) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Prediction')
        plt.close()

    def test_failed_plot(self, dataset: OSCD) -> None:
        single_band_dataset = type(dataset)(root=dataset.root, bands=('B01',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            x = single_band_dataset[0].copy()
            single_band_dataset.plot(x, suptitle='Test')

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            OSCD100(split='foo')
