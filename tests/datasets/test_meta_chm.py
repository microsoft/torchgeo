# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.


import os
import urllib.request
from collections.abc import Callable
from typing import BinaryIO

import matplotlib.pyplot as plt
import pytest
import torch
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import IntersectionDataset, MetaCHM, UnionDataset

pytest.importorskip('pyarrow')


class TestMetaCHM:
    @pytest.fixture
    def dataset(
        self, monkeypatch: MonkeyPatch, test_data: Callable[[str], str]
    ) -> MetaCHM:
        root = test_data('meta_chm')

        def urlopen(*args: object, **kwargs: object) -> BinaryIO:
            return open(os.path.join(root, 'items.parquet'), 'rb')

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
