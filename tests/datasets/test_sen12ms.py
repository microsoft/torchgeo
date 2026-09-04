# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn
from torch.utils.data import ConcatDataset

from torchgeo.datasets import SEN12MS, DatasetNotFoundError, RGBBandsMissingError


class TestSEN12MS:
    @pytest.fixture(params=['train', 'test'])
    def dataset(
        self,
        monkeypatch: MonkeyPatch,
        request: SubRequest,
        test_data: Callable[[str], str],
    ) -> SEN12MS:
        root = test_data('sen12ms')
        md5s = [
            hashlib.md5((Path(root) / filename).read_bytes()).hexdigest()
            for filename in SEN12MS.filenames
        ]
        monkeypatch.setattr(SEN12MS, 'md5s', md5s)
        split = request.param
        transforms = nn.Identity()
        return SEN12MS(root, split, transforms=transforms, checksum=True)

    def test_getitem(self, dataset: SEN12MS) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['image'].shape[0] == 15

    def test_len(self, dataset: SEN12MS) -> None:
        assert len(dataset) == 8

    def test_add(self, dataset: SEN12MS) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 16

    def test_out_of_bounds(self, dataset: SEN12MS) -> None:
        with pytest.raises(IndexError):
            dataset[8]

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SEN12MS(tmp_path, checksum=True)

        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SEN12MS(tmp_path, checksum=False)

    def test_check_integrity_light(self, test_data: Callable[[str], str]) -> None:
        root = test_data('sen12ms')
        ds = SEN12MS(root, checksum=False)
        assert isinstance(ds, SEN12MS)

    def test_band_subsets(self, test_data: Callable[[str], str]) -> None:
        root = test_data('sen12ms')
        for bands in SEN12MS.BAND_SETS.values():
            ds = SEN12MS(root, bands=bands, checksum=False)
            x = ds[0]['image']
            assert x.shape[0] == len(bands)

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            SEN12MS(bands=('OK', 'BK'))

    def test_plot(self, dataset: SEN12MS) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        sample = dataset[0]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='prediction')
        plt.close()

    def test_plot_rgb(self, dataset: SEN12MS) -> None:
        dataset = SEN12MS(root=dataset.root, bands=('B03',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            dataset.plot(dataset[0], suptitle='Single Band')
