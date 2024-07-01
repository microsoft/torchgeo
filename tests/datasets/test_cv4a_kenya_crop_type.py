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
    CV4AKenyaCropType,
    DatasetNotFoundError,
    RGBBandsMissingError,
)
from torchgeo.datasets.utils import Executable


class TestCV4AKenyaCropType:
    @pytest.fixture
    def dataset(
        self, azcopy: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> CV4AKenyaCropType:
        url = os.path.join('tests', 'data', 'cv4a_kenya_crop_type')
        monkeypatch.setattr(CV4AKenyaCropType, 'url', url)
        monkeypatch.setattr(CV4AKenyaCropType, 'tiles', list(map(str, range(1))))
        monkeypatch.setattr(CV4AKenyaCropType, 'dates', ['20190606'])
        monkeypatch.setattr(CV4AKenyaCropType, 'tile_height', 2)
        monkeypatch.setattr(CV4AKenyaCropType, 'tile_width', 2)
        root = str(tmp_path)
        transforms = nn.Identity()
        return CV4AKenyaCropType(root, transforms=transforms, download=True)

    def test_getitem(self, dataset: CV4AKenyaCropType) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        assert isinstance(x['x'], torch.Tensor)
        assert isinstance(x['y'], torch.Tensor)

    def test_len(self, dataset: CV4AKenyaCropType) -> None:
        assert len(dataset) == 1

    def test_add(self, dataset: CV4AKenyaCropType) -> None:
        ds = dataset + dataset
        assert isinstance(ds, ConcatDataset)
        assert len(ds) == 2

    def test_already_downloaded(self, dataset: CV4AKenyaCropType) -> None:
        CV4AKenyaCropType(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            CV4AKenyaCropType(str(tmp_path))

    def test_invalid_bands(self) -> None:
        with pytest.raises(AssertionError):
            CV4AKenyaCropType(bands=('foo', 'bar'))

    def test_plot(self, dataset: CV4AKenyaCropType) -> None:
        sample = dataset[0]
        dataset.plot(sample, time_step=0, suptitle='Test')
        plt.close()
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, time_step=0, suptitle='Pred')
        plt.close()

    def test_plot_rgb(self, dataset: CV4AKenyaCropType) -> None:
        dataset = CV4AKenyaCropType(root=dataset.root, bands=tuple(['B01']))
        match = 'Dataset does not contain some of the RGB bands'
        with pytest.raises(RGBBandsMissingError, match=match):
            dataset.plot(dataset[0])
