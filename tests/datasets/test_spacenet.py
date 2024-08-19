# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, SpaceNet1
from torchgeo.datasets.utils import Executable


class TestSpaceNet:
    @pytest.fixture
    def dataset(
        self, aws: Executable, monkeypatch: MonkeyPatch, tmp_path: Path
    ) -> SpaceNet1:
        url = os.path.join(
            'tests', 'data', 'spacenet', '{dataset_id}', 'train', '{tarball}'
        )
        monkeypatch.setattr(SpaceNet1, 'url', url)
        transforms = nn.Identity()
        return SpaceNet1(tmp_path, transforms=transforms, download=True)

    @pytest.mark.parametrize('index', [0, 1])
    def test_getitem(self, dataset: SpaceNet1, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: SpaceNet1) -> None:
        assert len(dataset) == 3

    def test_already_extracted(self, dataset: SpaceNet1) -> None:
        SpaceNet1(root=dataset.root)

    def test_already_downloaded(self, dataset: SpaceNet1) -> None:
        for product in ['3band', '8band', 'geojson']:
            dir = os.path.join(dataset.root, dataset.dataset_id, dataset.split, product)
            shutil.rmtree(dir)
        SpaceNet1(root=dataset.root)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SpaceNet1(tmp_path)

    def test_plot(self, dataset: SpaceNet1) -> None:
        x = dataset[0]
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask']
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_image_id(self, monkeypatch: MonkeyPatch, dataset: SpaceNet1) -> None:
        file_regex = r'global_monthly_(\d+.*\d+)'
        monkeypatch.setattr(dataset, 'file_regex', file_regex)
        dataset._image_id('global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160.tif')

    def test_list_files(self, monkeypatch: MonkeyPatch, dataset: SpaceNet1) -> None:
        directory_glob = os.path.join('**', 'AOI_{aoi}_*', '{product}')
        monkeypatch.setattr(dataset, 'directory_glob', directory_glob)
        dataset._list_files(aoi=1)
