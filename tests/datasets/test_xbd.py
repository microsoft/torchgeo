# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch
from torch import nn

from torchgeo.datasets import DatasetNotFoundError, xBD, xBDDistShift


class TestxBD:
    @pytest.fixture(params=product([xBD, xBDDistShift], ['train', 'test']))
    def dataset(self, monkeypatch: MonkeyPatch, request: SubRequest) -> xBD:
        base_class: type[xBD] = request.param[0]
        split = request.param[1]
        monkeypatch.setattr(
            base_class,
            'metadata',
            {
                'train': {
                    'filename': 'train_images_labels_targets.tar.gz',
                    'directory': 'train',
                },
                'test': {
                    'filename': 'test_images_labels_targets.tar.gz',
                    'directory': 'test',
                },
            },
        )
        root = os.path.join('tests', 'data', 'xbd')
        transforms = nn.Identity()
        if base_class is xBDDistShift:
            return base_class(
                root=root,
                split=split,
                id_disaster='hurricane-harvey',
                ood_disaster='hurricane-michael',
                transforms=transforms,
            )
        return base_class(root=root, split=split, transforms=transforms)

    def test_getitem(self, dataset: xBD) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)
        if isinstance(dataset, xBDDistShift):
            assert x['image'].ndim == 3
            assert set(torch.unique(x['mask']).tolist()) <= {0, 1}
            disaster = (
                'hurricane-harvey' if dataset.split == 'train' else 'hurricane-michael'
            )
            assert disaster in dataset.files[0]['image']
        else:
            assert x['image'].ndim == 4

    def test_len(self, dataset: xBD) -> None:
        assert len(dataset) == 2

    def test_extract(self, tmp_path: Path) -> None:
        shutil.copyfile(
            os.path.join('tests', 'data', 'xbd', 'train_images_labels_targets.tar.gz'),
            os.path.join(tmp_path, 'train_images_labels_targets.tar.gz'),
        )
        shutil.copyfile(
            os.path.join('tests', 'data', 'xbd', 'test_images_labels_targets.tar.gz'),
            os.path.join(tmp_path, 'test_images_labels_targets.tar.gz'),
        )
        xBD(root=tmp_path, checksum=False)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(
            os.path.join(tmp_path, 'train_images_labels_targets.tar.gz'), 'w'
        ) as f:
            f.write('bad')
        with open(
            os.path.join(tmp_path, 'test_images_labels_targets.tar.gz'), 'w'
        ) as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            xBD(root=tmp_path, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            xBD(tmp_path)

    def test_plot(self, dataset: xBD) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask']
        dataset.plot(x)
        plt.close()

    def test_pre_post_both(self) -> None:
        dataset = xBDDistShift(
            root=os.path.join('tests', 'data', 'xbd'),
            split='train',
            id_disaster='hurricane-harvey',
            id_pre_post='both',
            ood_disaster='hurricane-michael',
        )
        assert len(dataset) == 4

    def test_default_configuration(self) -> None:
        dataset = xBDDistShift(root=os.path.join('tests', 'data', 'xbd'))
        assert dataset.id_disaster == 'hurricane-matthew'
        assert dataset.ood_disaster == 'mexico-earthquake'
