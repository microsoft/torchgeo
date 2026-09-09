# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import pytest
import torch
from _pytest.fixtures import SubRequest
from torch import nn

from torchgeo.datasets import DatasetNotFoundError, VisDrone


class TestVisDrone:
    @pytest.fixture(params=['train', 'val', 'test'])
    def dataset(self, request: SubRequest) -> VisDrone:
        root = os.path.join('tests', 'data', 'visdrone')
        return VisDrone(root, request.param, transforms=nn.Identity())

    def test_getitem(self, dataset: VisDrone) -> None:
        sample = dataset[0]
        assert sample['image'].shape == (3, 8, 8)
        assert torch.equal(sample['bbox_xyxy'], torch.tensor([[1, 2, 4, 6]]))
        assert torch.equal(sample['label'], torch.tensor([0]))

    def test_len(self, dataset: VisDrone) -> None:
        assert len(dataset) == 1

    def test_invalid_split(self) -> None:
        with pytest.raises(AssertionError):
            VisDrone(split='invalid')  # ty: ignore[invalid-argument-type]

    def test_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            VisDrone(tmp_path)
