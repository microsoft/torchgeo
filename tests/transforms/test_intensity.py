# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import pytest
import torch
from torch import Tensor

from torchgeo.transforms import Sentinel1ChangeMap, ToDecibelScale


@pytest.fixture
def sample() -> dict[str, Tensor]:
    return {
        'image': torch.arange(4 * 4 * 4, dtype=torch.float).view(4, 4, 4),
        'mask': torch.arange(4 * 4, dtype=torch.long).view(1, 4, 4),
    }


@pytest.fixture
def batch() -> dict[str, Tensor]:
    return {
        'image': torch.arange(2 * 4 * 4 * 4, dtype=torch.float).view(2, 4, 4, 4),
        'mask': torch.arange(2 * 4 * 4, dtype=torch.long).view(2, 1, 4, 4),
    }


def test_to_decibel_scale(sample: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(ToDecibelScale(), keepdim=True, data_keys=None)
    output = aug(sample)
    assert output['image'].shape == sample['image'].shape
    assert output['mask'].shape == sample['mask'].shape


def test_to_decibel_scale_batch(batch: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(ToDecibelScale(), keepdim=True, data_keys=None)
    output = aug(batch)
    assert output['image'].shape == batch['image'].shape
    assert output['mask'].shape == batch['mask'].shape


def test_sentinel1_change_map(sample: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(Sentinel1ChangeMap(), keepdim=True, data_keys=None)
    output = aug(sample)
    c, h, w = sample['image'].shape
    assert output['image'].shape == (c // 2, h, w)
    assert output['mask'].shape == sample['mask'].shape


def test_sentinel1_change_map_batch(batch: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(Sentinel1ChangeMap(), keepdim=True, data_keys=None)
    output = aug(batch)
    b, c, h, w = batch['image'].shape
    assert output['image'].shape == (b, c // 2, h, w)
    assert output['mask'].shape == batch['mask'].shape
