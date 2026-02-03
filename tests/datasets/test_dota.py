# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch

from torchgeo.datasets import DOTA, DatasetNotFoundError


class TestDOTA:
    @pytest.fixture(
        params=product(
            ['train', 'val'], ['1.0', '1.5', '2.0'], ['horizontal', 'oriented']
        )
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> DOTA:
        url = os.path.join('tests', 'data', 'dota', '{}')
        monkeypatch.setattr(DOTA, 'url', url)

        file_info = {
            'train': {
                'images': {
                    '1.0': {'filename': 'dotav1.0_images_train.tar.gz', 'md5': ''},
                    '1.5': {'filename': 'dotav1.0_images_train.tar.gz', 'md5': ''},
                    '2.0': {'filename': 'dotav2.0_images_train.tar.gz', 'md5': ''},
                },
                'annotations': {
                    '1.0': {'filename': 'dotav1.0_annotations_train.tar.gz', 'md5': ''},
                    '1.5': {'filename': 'dotav1.5_annotations_train.tar.gz', 'md5': ''},
                    '2.0': {'filename': 'dotav2.0_annotations_train.tar.gz', 'md5': ''},
                },
            },
            'val': {
                'images': {
                    '1.0': {'filename': 'dotav1.0_images_val.tar.gz', 'md5': ''},
                    '1.5': {'filename': 'dotav1.0_images_val.tar.gz', 'md5': ''},
                    '2.0': {'filename': 'dotav2.0_images_val.tar.gz', 'md5': ''},
                },
                'annotations': {
                    '1.0': {'filename': 'dotav1.0_annotations_val.tar.gz', 'md5': ''},
                    '1.5': {'filename': 'dotav1.5_annotations_val.tar.gz', 'md5': ''},
                    '2.0': {'filename': 'dotav2.0_annotations_val.tar.gz', 'md5': ''},
                },
            },
        }
        monkeypatch.setattr(DOTA, 'file_info', file_info)

        root = tmp_path
        split, version, bbox_orientation = request.param

        transforms = nn.Identity()

        return DOTA(
            root,
            split,
            version=version,
            bbox_orientation=bbox_orientation,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_getitem(self, dataset: DOTA) -> None:
        for i in range(len(dataset)):
            x = dataset[i]
            assert isinstance(x, dict)
            assert isinstance(x['image'], torch.Tensor)
            assert isinstance(x['labels'], torch.Tensor)
            if dataset.bbox_orientation == 'oriented':
                bbox_key = 'bbox'
            else:
                bbox_key = 'bbox_xyxy'
            assert isinstance(x[bbox_key], torch.Tensor)

            if dataset.bbox_orientation == 'oriented':
                assert x[bbox_key].shape[1] == 8
            else:
                assert x[bbox_key].shape[1] == 4

            assert x['labels'].shape[0] == x[bbox_key].shape[0]

    def test_len(self, dataset: DOTA) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 2

    def test_already_downloaded(self, dataset: DOTA) -> None:
        DOTA(root=dataset.root, download=True)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        files = [
            'dotav1.0_images_train.tar.gz',
            'dotav1.0_annotations_train.tar.gz',
            'dotav1.5_annotations_train.tar.gz',
            'dotav1.5_annotations_val.tar.gz',
            'dotav1.0_images_val.tar.gz',
            'dotav1.0_annotations_val.tar.gz',
            'dotav2.0_images_train.tar.gz',
            'dotav2.0_annotations_train.tar.gz',
            'dotav2.0_images_val.tar.gz',
            'dotav2.0_annotations_val.tar.gz',
            'samples.csv',
        ]
        for path in files:
            shutil.copyfile(
                os.path.join('tests', 'data', 'dota', path),
                os.path.join(str(tmp_path), path),
            )

        DOTA(root=tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'dotav1.0_images_train.tar.gz'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Archive'):
            DOTA(root=tmp_path, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DOTA(tmp_path)

    def test_plot(self, dataset: DOTA) -> None:
        x = dataset[1]
        dataset.plot(x, suptitle='Test')
        plt.close()
