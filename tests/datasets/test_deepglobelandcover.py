# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest

from torchgeo.datasets import DatasetNotFoundError, DeepGlobeLandCover


class TestDeepGlobeLandCover:
    @pytest.fixture(params=['train', 'test'])
    def dataset(self, request: SubRequest) -> DeepGlobeLandCover:
        root = os.path.join('tests', 'data', 'deepglobelandcover')
        split = request.param
        transforms = nn.Identity()
        return DeepGlobeLandCover(root, split, transforms)

    def test_getitem(self, dataset: DeepGlobeLandCover) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['mask'], torch.Tensor)

    def test_len(self, dataset: DeepGlobeLandCover) -> None:
        assert len(dataset) == 3

    def test_extract(self, tmp_path: Path) -> None:
        root = os.path.join('tests', 'data', 'deepglobelandcover')
        filename = 'data.zip'
        shutil.copyfile(os.path.join(root, filename), os.path.join(tmp_path, filename))
        DeepGlobeLandCover(root=tmp_path)

    def test_corrupted(self, tmp_path: Path) -> None:
        with open(os.path.join(tmp_path, 'data.zip'), 'w') as f:
            f.write('bad')
        with pytest.raises(RuntimeError, match='Dataset found, but corrupted'):
            DeepGlobeLandCover(root=tmp_path, checksum=True)

    def test_not_downloaded(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            DeepGlobeLandCover(tmp_path)

    def test_plot(self, dataset: DeepGlobeLandCover) -> None:
        x = dataset[0].copy()
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
        x['prediction'] = x['mask'].clone()
        dataset.plot(x)
        plt.close()
