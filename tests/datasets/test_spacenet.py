# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DatasetNotFoundError, SpaceNet, SpaceNet1, SpaceNet6
from torchgeo.datasets.utils import Executable


class TestSpaceNet:
    @pytest.fixture(params=[SpaceNet1, SpaceNet6])
    def dataset(
        self,
        request: SubRequest,
        aws: Executable,
        monkeypatch: MonkeyPatch,
        tmp_path: Path,
    ) -> SpaceNet:
        dataset_class: type[SpaceNet] = request.param
        url = os.path.join(
            'tests',
            'data',
            'spacenet',
            dataset_class.__name__.lower(),
            '{dataset_id}',
            'train',
            '{tarball}',
        )
        monkeypatch.setattr(dataset_class, 'url', url)
        transforms = nn.Identity()
        return dataset_class(tmp_path, transforms=transforms, download=True)

    @pytest.mark.parametrize('index', [0, 1])
    def test_getitem(self, dataset: SpaceNet, index: int) -> None:
        x = dataset[index]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: SpaceNet) -> None:
        assert len(dataset) == 4

    def test_already_extracted(self, dataset: SpaceNet) -> None:
        dataset.__class__(root=dataset.root)

    def test_already_downloaded(self, dataset: SpaceNet) -> None:
        if dataset.dataset_id == 'SN1_buildings':
            base_dir = os.path.join(dataset.root, dataset.dataset_id, dataset.split)
        elif dataset.dataset_id == 'SN6_buildings':
            base_dir = os.path.join(
                dataset.root,
                dataset.dataset_id,
                dataset.split,
                dataset.split,
                'AOI_11_Rotterdam',
            )
        for product in dataset.valid_images['train'] + list(dataset.valid_masks):
            dir = os.path.join(base_dir, product)
            shutil.rmtree(dir)
        dataset.__class__(root=dataset.root)

    def test_not_downloaded(self, tmp_path: Path, dataset: SpaceNet) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            dataset.__class__(root=os.path.join(tmp_path, 'dummy'))

    def test_plot(self, dataset: SpaceNet) -> None:
        x = dataset[0]
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask']
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_image_id(self, monkeypatch: MonkeyPatch, dataset: SpaceNet) -> None:
        file_regex = r'global_monthly_(\d+.*\d+)'
        monkeypatch.setattr(dataset, 'file_regex', file_regex)
        dataset._image_id('global_monthly_2018_01_mosaic_L15-0331E-1257N_1327_3160.tif')

    def test_list_files(self, monkeypatch: MonkeyPatch, dataset: SpaceNet) -> None:
        directory_glob = os.path.join('**', 'AOI_{aoi}_*', '{product}')
        monkeypatch.setattr(dataset, 'directory_glob', directory_glob)
        dataset._list_files(aoi=1)
