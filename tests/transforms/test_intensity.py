# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import pytest
import torch
from torch import Tensor

from torchgeo.transforms import PowerToDecibel, ToThresholdedChangeMask


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


def test_power_to_decibel(sample: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(PowerToDecibel(), keepdim=True, data_keys=None)
    output = aug(sample)
    assert output['image'].shape == sample['image'].shape
    assert output['mask'].shape == sample['mask'].shape


def test_power_to_decibel_batch(batch: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(PowerToDecibel(), keepdim=True, data_keys=None)
    output = aug(batch)
    assert output['image'].shape == batch['image'].shape
    assert output['mask'].shape == batch['mask'].shape


def test_thresholded_change_mask(sample: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(
        ToThresholdedChangeMask(), keepdim=True, data_keys=None
    )
    output = aug(sample)
    c, h, w = sample['image'].shape
    assert output['image'].shape == (c // 2, h, w)
    assert output['mask'].shape == sample['mask'].shape


def test_thresholded_change_mask_batch(batch: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(
        ToThresholdedChangeMask(), keepdim=True, data_keys=None
    )
    output = aug(batch)
    b, c, h, w = batch['image'].shape
    assert output['image'].shape == (b, c // 2, h, w)
    assert output['mask'].shape == batch['mask'].shape
