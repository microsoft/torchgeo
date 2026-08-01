# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import DatasetNotFoundError, TimeSen2Crop


class TestTimeSen2Crop:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> TimeSen2Crop:
        md5 = 'f109ac2b09002187d2a05288f6dfaa56'
        monkeypatch.setattr(TimeSen2Crop, 'md5', md5)
        url = os.path.join('tests', 'data', 'timesen2crop', 'TimeSen2Crop.zip')
        monkeypatch.setattr(TimeSen2Crop, 'url', url)
        root = tmp_path
        transforms = nn.Identity()
        return TimeSen2Crop(root, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: TimeSen2Crop) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['condition'], torch.Tensor)
        assert isinstance(x['date'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].shape == (5, len(dataset.all_bands))
        assert x['condition'].shape == (5,)
        assert x['date'].shape == (5,)
        assert x['label'].ndim == 0

    def test_len(self, dataset: TimeSen2Crop) -> None:
        assert len(dataset) == 480

    def test_bands(self, dataset: TimeSen2Crop) -> None:
        ds = TimeSen2Crop(root=dataset.root, bands=('B04', 'B03', 'B02'))
        x = ds[0]
        assert x['image'].shape == (5, 3)

    def test_2019_tile(self, dataset: TimeSen2Crop) -> None:
        ds = TimeSen2Crop(root=dataset.root, tiles=('2019_33UVP',))
        assert len(ds) == 32
        x = ds[0]
        assert x['date'][0] == 20181008

    def test_invalid_tiles(self, dataset: TimeSen2Crop) -> None:
        with pytest.raises(AssertionError, match='Only the following tiles'):
            TimeSen2Crop(root=dataset.root, tiles=('33ZZZ',))

    def test_invalid_bands(self, dataset: TimeSen2Crop) -> None:
        with pytest.raises(AssertionError, match='Only the following bands'):
            TimeSen2Crop(root=dataset.root, bands=('B01',))

    def test_already_downloaded(self, dataset: TimeSen2Crop) -> None:
        TimeSen2Crop(root=dataset.root, download=True)

    def test_already_downloaded_not_extracted(
        self, dataset: TimeSen2Crop, tmp_path: Path
    ) -> None:
        shutil.rmtree(os.path.join(tmp_path, dataset.directory))
        TimeSen2Crop(root=tmp_path, download=False)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            TimeSen2Crop(tmp_path)

    def test_plot(self, dataset: TimeSen2Crop) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()
