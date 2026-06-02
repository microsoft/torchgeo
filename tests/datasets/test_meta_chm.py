# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import urllib.request
from typing import BinaryIO

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import IntersectionDataset, MetaCHM, UnionDataset

pytest.importorskip('pyarrow')

ROOT = os.path.join('tests', 'data', 'meta_chm')


class TestMetaCHM:
    @pytest.fixture
    def dataset(self, monkeypatch: MonkeyPatch) -> MetaCHM:
        def urlopen(*args: object, **kwargs: object) -> BinaryIO:
            return open(os.path.join(ROOT, 'items.parquet'), 'rb')

        monkeypatch.setattr(urllib.request, 'urlopen', urlopen)
        return MetaCHM(transforms=nn.Identity())

    def test_getitem(self, dataset: MetaCHM) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['mask'], torch.Tensor)
        assert x['mask'].dtype == torch.float32

    def test_len(self, dataset: MetaCHM) -> None:
        assert len(dataset) == 2

    def test_and(self, dataset: MetaCHM) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: MetaCHM) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: MetaCHM) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_plot_prediction(self, dataset: MetaCHM) -> None:
        x = dataset[dataset.bounds]
        x['prediction'] = x['mask'].clone()
        dataset.plot(x, suptitle='Prediction')
        plt.close()
