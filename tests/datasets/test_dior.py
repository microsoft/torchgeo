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

from torchgeo.datasets import DIOR, DatasetNotFoundError


class TestDIOR:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> DIOR:
        url = os.path.join('tests', 'data', 'dior', '{}')
        monkeypatch.setattr(DIOR, 'url', url)

        files = {
            'trainval': {
                'images': {
                    'filename': 'Images_trainval.zip',
                    'md5': '585e21ddd28fdf1166e463db43cfe68d',
                },
                'labels': {
                    'filename': 'Annotations_trainval.zip',
                    'md5': 'dcc93fa421804515029a5a574f34fafc',
                },
            },
            'test': {
                'images': {
                    'filename': 'Images_test.zip',
                    'md5': '0273920291def20cec60849b35eca713',
                }
            },
        }
        monkeypatch.setattr(DIOR, 'files', files)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return DIOR(
            root=root, split=split, transforms=transforms, download=True, checksum=True
        )

    def test_already_downloaded(self, dataset: DIOR) -> None:
        DIOR(root=dataset.root, download=True)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        files = [
            'Images_trainval.zip',
            'Annotations_trainval.zip',
            'Images_test.zip',
            'sample_df.csv',
        ]
        for path in files:
            shutil.copyfile(
                os.path.join('tests', 'data', 'dior', path),
                os.path.join(str(tmp_path), path),
            )

        DIOR(root=tmp_path)

    def test_getitem(self, dataset: DIOR) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert x['image'].shape[0] == 3
        assert x['image'].ndim == 3
        assert isinstance(x['image'], torch.Tensor)
        if dataset.split != 'test':
            assert isinstance(x['label'], torch.Tensor)
            assert isinstance(x['bbox_xyxy'], torch.Tensor)

    def test_len(self, dataset: DIOR) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 4
        else:
            assert len(dataset) == 2

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'Images_trainval.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            DIOR(root=tmp_path, checksum=True)

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DIOR(tmp_path)

    def test_plot(self, dataset: DIOR) -> None:
        if dataset.split != 'test':
            x = dataset[0].copy()
            dataset.plot(x, suptitle='Test')
            plt.close()
