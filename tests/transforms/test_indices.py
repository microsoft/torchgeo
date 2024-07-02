# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import pytest
import torch
from torch import Tensor

from torchgeo.transforms import (
    AppendBNDVI,
    AppendGBNDVI,
    AppendGNDVI,
    AppendGRNDVI,
    AppendNBR,
    AppendNDBI,
    AppendNDRE,
    AppendNDSI,
    AppendNDVI,
    AppendNDWI,
    AppendNormalizedDifferenceIndex,
    AppendRBNDVI,
    AppendSWI,
    AppendTriBandNormalizedDifferenceIndex,
)


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


def test_append_index_sample(sample: dict[str, Tensor]) -> None:
    c, h, w = sample['image'].shape
    aug = K.AugmentationSequential(
        AppendNormalizedDifferenceIndex(index_a=0, index_b=1), data_keys=None
    )
    output = aug(sample)
    assert output['image'].shape == (1, c + 1, h, w)


def test_append_index_batch(batch: dict[str, Tensor]) -> None:
    b, c, h, w = batch['image'].shape
    aug = K.AugmentationSequential(
        AppendNormalizedDifferenceIndex(index_a=0, index_b=1), data_keys=None
    )
    output = aug(batch)
    assert output['image'].shape == (b, c + 1, h, w)


def test_append_triband_index_batch(batch: dict[str, Tensor]) -> None:
    b, c, h, w = batch['image'].shape
    aug = K.AugmentationSequential(
        AppendTriBandNormalizedDifferenceIndex(index_a=0, index_b=1, index_c=2),
        data_keys=None,
    )
    output = aug(batch)
    assert output['image'].shape == (b, c + 1, h, w)


@pytest.mark.parametrize(
    'index',
    [
        AppendBNDVI,
        AppendNBR,
        AppendNDBI,
        AppendNDRE,
        AppendNDSI,
        AppendNDVI,
        AppendNDWI,
        AppendSWI,
        AppendGNDVI,
    ],
)
def test_append_normalized_difference_indices(
    sample: dict[str, Tensor], index: AppendNormalizedDifferenceIndex
) -> None:
    c, h, w = sample['image'].shape
    aug = K.AugmentationSequential(index(0, 1), data_keys=None)
    output = aug(sample)
    assert output['image'].shape == (1, c + 1, h, w)


@pytest.mark.parametrize('index', [AppendGBNDVI, AppendGRNDVI, AppendRBNDVI])
def test_append_tri_band_normalized_difference_indices(
    sample: dict[str, Tensor], index: AppendTriBandNormalizedDifferenceIndex
) -> None:
    c, h, w = sample['image'].shape
    aug = K.AugmentationSequential(index(0, 1, 2), data_keys=None)
    output = aug(sample)
    assert output['image'].shape == (1, c + 1, h, w)
