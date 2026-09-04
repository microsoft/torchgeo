# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from _pytest.fixtures import SubRequest
from torch import nn

from torchgeo.datasets import DatasetNotFoundError, InriaAerialImageLabeling


class TestInriaAerialImageLabeling:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(
        self, request: SubRequest, test_data: Callable[[str], str]
    ) -> InriaAerialImageLabeling:
        root = Path(test_data('inria'))
        transforms = nn.Identity()
        return InriaAerialImageLabeling(
            root, split=request.param, transforms=transforms, checksum=False
        )

    def test_getitem(self, dataset: InriaAerialImageLabeling) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        if dataset.split == 'train':
            assert isinstance(x['mask'], torch.Tensor)
            assert x['mask'].ndim == 2
        assert x['image'].shape[0] == 3
        assert x['image'].ndim == 3

    def test_len(self, dataset: InriaAerialImageLabeling) -> None:
        if dataset.split == 'train':
            assert len(dataset) == 2
        elif dataset.split == 'val':
            assert len(dataset) == 5
        elif dataset.split == 'test':
            assert len(dataset) == 7

    def test_already_downloaded(self, dataset: InriaAerialImageLabeling) -> None:
        InriaAerialImageLabeling(root=dataset.root)

    def test_not_downloaded(self, tmp_path: str) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            InriaAerialImageLabeling(tmp_path)

    def test_dataset_checksum(self, tmp_path: Path) -> None:
        (tmp_path / InriaAerialImageLabeling.filename).touch()
        with pytest.raises(RuntimeError, match='Dataset corrupted'):
            InriaAerialImageLabeling(root=tmp_path, checksum=True)

    def test_extract(self, tmp_path: Path, test_data: Callable[[str], str]) -> None:
        src = Path(os.path.join(test_data('inria'), 'NEW2-AerialImageDataset.zip'))
        shutil.copy(src, tmp_path)
        InriaAerialImageLabeling(tmp_path, checksum=False)

    def test_plot(self, dataset: InriaAerialImageLabeling) -> None:
        x = dataset[0].copy()
        if dataset.split == 'train':
            x['prediction'] = x['mask']
        dataset.plot(x, suptitle='Test')
        plt.close()
        dataset.plot(x, show_titles=False)
        plt.close()
