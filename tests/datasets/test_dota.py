# Copyright (c) Microsoft Corporation. All rights reserved.
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
    @pytest.fixture(params=product(['train', 'val'], ['1.0', '2.0']))
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> DOTA:
        url = os.path.join('tests', 'data', 'dota', '{}')
        monkeypatch.setattr(DOTA, 'url', url)

        file_info = {
            'train': {
                'images': {
                    '1.0': {
                        'filename': 'dotav1_images_train.tar.gz',
                        'md5': '14296c11c897cb7718558815a2b1bf69',
                    },
                    '2.0': {
                        'filename': 'dotav2_images_train.tar.gz',
                        'md5': 'fc80227b1f9b99cf5a7a3d0c5798efd0',
                    },
                },
                'annotations': {
                    '1.0': {
                        'filename': 'dotav1_annotations_train.tar.gz',
                        'md5': '805dc01688e00895a594c637569a2e1a',
                    },
                    '2.0': {
                        'filename': 'dotav2_annotations_train.tar.gz',
                        'md5': '723bceb26bc52a5de45902fada335c36',
                    },
                },
            },
            'val': {
                'images': {
                    '1.0': {
                        'filename': 'dotav1_images_val.tar.gz',
                        'md5': 'a95acf48281b7fc800666974730aeffd',
                    },
                    '2.0': {
                        'filename': 'dotav2_images_val.tar.gz',
                        'md5': '7c4ebb3317f970b26de273cd7313d46f',
                    },
                },
                'annotations': {
                    '1.0': {
                        'filename': 'dotav1_annotations_val.tar.gz',
                        'md5': '435a4a77c62eff955dd30a1b2a13894f',
                    },
                    '2.0': {
                        'filename': 'dotav2_annotations_val.tar.gz',
                        'md5': '86b629c6c8a1d924841d34de4eeb87ec',
                    },
                },
            },
        }
        monkeypatch.setattr(DOTA, 'file_info', file_info)

        root = tmp_path
        split, version = request.param
        if version == '2.0':
            bbox_orientation = 'obb'
        else:
            bbox_orientation = 'hbb'

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
            assert isinstance(x['boxes'], torch.Tensor)

            if dataset.bbox_orientation == 'obb':
                assert x['boxes'].shape[1] == 8
            else:
                assert x['boxes'].shape[1] == 4

            assert x['labels'].shape[0] == x['boxes'].shape[0]

    def test_len(self, dataset: DOTA) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 2

    def test_already_downloaded(self, dataset: DOTA) -> None:
        DOTA(root=dataset.root, download=True)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        files = [
            'dotav1_images_train.tar.gz',
            'dotav1_annotations_train.tar.gz',
            'dotav1_images_val.tar.gz',
            'dotav1_annotations_val.tar.gz',
            'dotav2_images_train.tar.gz',
            'dotav2_annotations_train.tar.gz',
            'dotav2_images_val.tar.gz',
            'dotav2_annotations_val.tar.gz',
            'samples.parquet',
        ]
        for path in files:
            shutil.copyfile(
                os.path.join('tests', 'data', 'dota', path),
                os.path.join(str(tmp_path), path),
            )

        DOTA(root=tmp_path)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            DOTA(split='foo')

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'dotav1_images_train.tar.gz'), 'w') as f:
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
