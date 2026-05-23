# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from torchgeo.datasets import DatasetNotFoundError, RGBBandsMissingError, So2Sat

pytest.importorskip('h5py', minversion='3.10')


class TestSo2Sat:
    @pytest.fixture(params=['train', 'validation', 'test'])
    def dataset(self, request: SubRequest) -> So2Sat:
        root = os.path.join('tests', 'data', 'so2sat')
        split = request.param
        transforms = nn.Identity()
        return So2Sat(root=root, split=split, transforms=transforms)

    def test_getitem(self, dataset: So2Sat) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)

    def test_len(self, dataset: So2Sat) -> None:
        assert len(dataset) == 2

    def test_out_of_bounds(self, dataset: So2Sat) -> None:
        with pytest.raises(IndexError):
            dataset[2]

    def test_invalid_bands(self) -> None:
        with pytest.raises(ValueError):
            So2Sat(bands=('OK', 'BK'))

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            So2Sat(tmp_path)

    def test_plot(self, dataset: So2Sat) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['label'].clone()
        dataset.plot(x)
        plt.close()

    def test_plot_rgb(self, dataset: So2Sat) -> None:
        dataset = So2Sat(root=dataset.root, bands=('S2_B03',))
        with pytest.raises(
            RGBBandsMissingError, match='Dataset does not contain some of the RGB bands'
        ):
            dataset.plot(dataset[0], suptitle='Single Band')
