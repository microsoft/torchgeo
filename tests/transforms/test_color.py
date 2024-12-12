# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import pytest
import torch
from torch import Tensor

from torchgeo.transforms import RandomGrayscale


@pytest.fixture
def sample() -> dict[str, Tensor]:
    return {
        'image': torch.arange(3 * 4 * 4, dtype=torch.float).view(3, 4, 4),
        'mask': torch.arange(4 * 4, dtype=torch.long).view(1, 4, 4),
    }


@pytest.fixture
def batch() -> dict[str, Tensor]:
    return {
        'image': torch.arange(2 * 3 * 4 * 4, dtype=torch.float).view(2, 3, 4, 4),
        'mask': torch.arange(2 * 4 * 4, dtype=torch.long).view(2, 1, 4, 4),
    }


@pytest.mark.parametrize(
    'weights',
    [
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0.299, 0.587, 0.114]),
        torch.tensor([1.0, 2.0, 3.0]),
    ],
)
def test_random_grayscale_sample(weights: Tensor, sample: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(
        RandomGrayscale(weights, p=1), keepdim=True, data_keys=None
    )
    output = aug(sample)
    assert output['image'].shape == sample['image'].shape
    for i in range(1, 3):
        assert torch.allclose(output['image'][0], output['image'][i])


@pytest.mark.parametrize(
    'weights',
    [
        torch.tensor([1.0, 1.0, 1.0]),
        torch.tensor([0.299, 0.587, 0.114]),
        torch.tensor([1.0, 2.0, 3.0]),
    ],
)
def test_random_grayscale_batch(weights: Tensor, batch: dict[str, Tensor]) -> None:
    aug = K.AugmentationSequential(RandomGrayscale(weights, p=1), data_keys=None)
    output = aug(batch)
    assert output['image'].shape == batch['image'].shape
    for i in range(1, 3):
        assert torch.allclose(output['image'][0, 0], output['image'][0, i])
