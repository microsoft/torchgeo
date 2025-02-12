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

from torchgeo.datasets import BRIGHTDFC2025, DatasetNotFoundError


class TestBRIGHTDFC2025:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, request: SubRequest
    ) -> BRIGHTDFC2025:
        url = os.path.join('tests', 'data', 'bright', 'dfc25_track2_trainval.zip')
        monkeypatch.setattr(BRIGHTDFC2025, 'url', url)
        root = tmp_path
        split = request.param
        transforms = nn.Identity()
        return BRIGHTDFC2025(root, split, transforms, download=True)

    def test_getitem(self, dataset: BRIGHTDFC2025) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image_pre'], torch.Tensor)
        assert x['image_pre'].shape[0] == 3
        assert isinstance(x['image_post'], torch.Tensor)
        assert x['image_post'].shape[0] == 3
        assert x['image_pre'].shape[-2:] == x['image_post'].shape[-2:]
        if dataset.split != 'test':
            assert isinstance(x['mask'], torch.Tensor)
            assert x['image_pre'].shape[-2:] == x['mask'].shape[-2:]

    def test_len(self, dataset: BRIGHTDFC2025) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 3
        elif dataset.split == 'val':
            assert len(dataset) == 1
        else:
            assert len(dataset) == 2

    def test_already_downloaded(self, dataset: BRIGHTDFC2025) -> None:
        BRIGHTDFC2025(root=dataset.root)

    def test_not_yet_extracted(self, tmp_path: Path) -> None:
        filename = 'dfc25_track2_trainval.zip'
        dir = os.path.join('tests', 'data', 'bright')
        shutil.copyfile(
            os.path.join(dir, filename), os.path.join(str(tmp_path), filename)
        )
        BRIGHTDFC2025(root=str(tmp_path))

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            BRIGHTDFC2025(split='foo')

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            BRIGHTDFC2025(tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'dfc25_track2_trainval.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted.'):
            BRIGHTDFC2025(root=tmp_path, checksum=True)

    def test_plot(self, dataset: BRIGHTDFC2025) -> None:
        dataset.plot(dataset[0], suptitle='Test')
        plt.close()

        if dataset.split != 'test':
            sample = dataset[0]
            sample['prediction'] = torch.clone(sample['mask'])
            dataset.plot(sample, suptitle='Prediction')
            plt.close()

            del sample['mask']
            dataset.plot(sample, suptitle='Only Prediction')
            plt.close()
