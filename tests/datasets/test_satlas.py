# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch.nn as nn
from pytest import MonkeyPatch
from torch import Tensor

from torchgeo.datasets import DatasetNotFoundError, SatlasPretrain
from torchgeo.datasets.utils import Executable


class TestSatlasPretrain:
    @pytest.fixture
    def dataset(
        self, aws: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SatlasPretrain:
        url = os.path.join('tests', 'data', 'satlas', '')
        monkeypatch.setattr(SatlasPretrain, 'url', url)
        images = ('landsat', 'naip', 'sentinel1', 'sentinel2')
        products = (*images, 'static', 'metadata')
        tarballs = {product: (f'{product}.tar',) for product in products}
        monkeypatch.setattr(SatlasPretrain, 'tarballs', tarballs)
        transforms = nn.Identity()
        return SatlasPretrain(
            tmp_path, images=images, transforms=transforms, download=True
        )

    @pytest.mark.parametrize('index', [0, 1])
    def test_getitem(self, dataset: SatlasPretrain, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        for image in dataset.images:
            assert isinstance(x[f'image_{image}'], Tensor)
            assert isinstance(x[f'time_{image}'], Tensor)
        for label in dataset.labels:
            assert isinstance(x[f'mask_{label}'], Tensor)

    def test_len(self, dataset: SatlasPretrain) -> None:
        assert len(dataset) == 2

    def test_already_downloaded(self, dataset: SatlasPretrain) -> None:
        shutil.rmtree(os.path.join(dataset.root, 'landsat'))
        SatlasPretrain(root=dataset.root, download=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SatlasPretrain(tmp_path)

    def test_plot(self, dataset: SatlasPretrain) -> None:
        x = dataset[0]
        x['prediction_land_cover'] = x['mask_land_cover']
        dataset.plot(x, suptitle='Test')
        plt.close()
