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

from torchgeo.datasets import SODAA, DatasetNotFoundError


class TestSODAA:
    @pytest.fixture(
        params=product(['train', 'val', 'test'], ['horizontal', 'oriented'])
    )
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> SODAA:
        url = os.path.join('tests', 'data', 'soda', '{}')
        monkeypatch.setattr(SODAA, 'url', url)
        files = {
            'images': {
                'filename': 'Images.zip',
                'md5sum': 'ed1128b2850932199f72b75bdc8bb863',
            },
            'labels': {
                'filename': 'Annotations.zip',
                'md5sum': '583256693eb4e87eb80d575a111d8857',
            },
        }
        monkeypatch.setattr(SODAA, 'files', files)
        split, bbox_orientation = request.param
        root = tmp_path
        transforms = nn.Identity()
        return SODAA(
            root=root,
            split=split,
            bbox_orientation=bbox_orientation,
            transforms=transforms,
            download=True,
            checksum=True,
        )

    def test_already_downloaded(self, dataset: SODAA) -> None:
        SODAA(root=dataset.root, download=True)

    def test_getitem(self, dataset: SODAA) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].shape[0] == 3
        assert x['image'].ndim == 3
        if dataset.bbox_orientation == 'horizontal':
            assert isinstance(x['boxes'], torch.Tensor)
        else:
            assert isinstance(x['boxes'], torch.Tensor)

    def test_len(self, dataset: SODAA) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        else:
            assert len(dataset) == 2

    def test_not_extracted(self, tmp_path: Path) -> None:
        files = ['Images.zip', 'Annotations.zip', 'sample_df.parquet']
        for file in files:
            shutil.copy(os.path.join('tests', 'data', 'soda', file), tmp_path)
        SODAA(root=tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'Images.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            SODAA(root=tmp_path, checksum=True)

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            SODAA(tmp_path)

    def test_plot(self, dataset: SODAA) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
