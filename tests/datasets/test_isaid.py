# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import matplotlib.pyplot as plt
from itertools import product
import pytest
import torch
import shutil
import torch.nn as nn
from _pytest.fixtures import SubRequest
from pytest import MonkeyPatch


from torchgeo.datasets import ISAID, DatasetNotFoundError

pytest.importorskip('pycocotools')


class TestISAID:
    @pytest.fixture(params=['train', 'val'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> ISAID:
        url = os.path.join('tests', 'data', 'isaid', '{}')
        monkeypatch.setattr(ISAID, 'img_url', url)
        monkeypatch.setattr(ISAID, 'label_url', url)

        img_url = 'https://huggingface.co/datasets/torchgeo/dota/tree/main/{}'

        img_files = {
            'train': {
                'filename': 'dotav1_images_train.tar.gz',
                'md5': 'a38ad9832066e2ca6d30b8eec65f9ce8',
            },
            'val': {
                'filename': 'dotav1_images_val.tar.gz',
                'md5': '154babe8091484bd85c6340f43cea1ea',
            },
        }

        label_files = {
            'train': {
                'filename': 'isaid_annotations_train.tar.gz',
                'md5': 'f4de0f6b38f1b11b121dc01c880aeb2a',
            },
            'val': {
                'filename': 'isaid_annotations_val.tar.gz',
                'md5': '88eccdf9744c201248266b9a784ffeab',
            },
        }

        root = tmp_path
        split = request.param

        transforms = nn.Identity()

        return ISAID(root, split, transforms=transforms, download=True, checksum=True)

    def test_getitem(self, dataset: ISAID) -> None:
        for i in range(len(dataset)):
            x = dataset[i]
            assert isinstance(x, dict)
            assert isinstance(x['image'], torch.Tensor)
            assert isinstance(x['masks'], torch.Tensor)
            assert isinstance(x['boxes'], torch.Tensor)

    def test_len(self, dataset: ISAID) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 2

    def test_already_downloaded(self, dataset: ISAID) -> None:
        ISAID(root=dataset.root, download=True)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        files = [
            'dotav1_images_train.tar.gz',
            'dotav1_images_val.tar.gz',
            'isaid_annotations_train.tar.gz',
            'isaid_annotations_val.tar.gz',
        ]
        for path in files:
            shutil.copyfile(
                os.path.join('tests', 'data', 'isaid', path),
                os.path.join(str(tmp_path), path),
            )

        ISAID(root=tmp_path)

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            ISAID(split='foo')

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'dotav1_images_train.tar.gz'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Archive'):
            ISAID(root=tmp_path, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            ISAID(tmp_path)
