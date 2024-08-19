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

from torchgeo.datasets import ChaBuD, DatasetNotFoundError

pytest.importorskip('h5py', minversion='3.6')


class TestChaBuD:
    @pytest.fixture(params=zip([ChaBuD.all_bands, ChaBuD.rgb_bands], ['train', 'val']))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> ChaBuD:
        data_dir = os.path.join('tests', 'data', 'chabud')
        url = os.path.join(data_dir, 'train_eval.hdf5')
        md5 = '1bec048beeb87a865c53f40ab418aa75'
        monkeypatch.setattr(ChaBuD, 'url', url)
        monkeypatch.setattr(ChaBuD, 'md5', md5)
        bands, split = request.param
        root = tmp_path
        transforms = nn.Identity()
        return ChaBuD(
            root=root,
            split=split,
            bands=bands,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: ChaBuD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

        # Image tests
        assert x['image'].ndim == 3

        if dataset.bands == ChaBuD.rgb_bands:
            assert x['image'].shape[0] == 2 * 3
        elif dataset.bands == ChaBuD.all_bands:
            assert x['image'].shape[0] == 2 * 12

        # Mask tests:
        assert x['mask'].ndim == 2

    def test_len(self, dataset: ChaBuD) -> None:
        assert len(dataset) == 4

    def test_already_downloaded(self, dataset: ChaBuD) -> None:
        ChaBuD(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ChaBuD(tmp_path)

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            ChaBuD(bands=['OK', 'BK'])

    def test_plot(self, dataset: ChaBuD) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='prediction')
        plt.close()

    def test_plot_rgb(self, dataset: ChaBuD) -> None:
        dataset = ChaBuD(root=dataset.root, bands=['B02'])
        with pytest.raises(ValueError, match="doesn't contain some of the RGB bands"):
            dataset.plot(dataset[0], suptitle='Single Band')
