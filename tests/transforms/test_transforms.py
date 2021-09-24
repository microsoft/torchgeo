# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict

import kornia.augmentation as K
import pytest
import torch
from torch import Tensor

from torchgeo.transforms import transforms


@pytest.fixture
def sample() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[0, 0, 2, 2], [1, 1, 3, 3]]
        ),
    }


@pytest.fixture
def batch() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
    }


def assert_matching(output: Dict[str, Tensor], expected: Dict[str, Tensor]) -> None:
    for key in expected:
        assert torch.allclose(output[key], expected[key])  # type: ignore[attr-defined]


def test_random_horizontal_flip(sample: Dict[str, Tensor]) -> None:
    tr = transforms.RandomHorizontalFlip(p=1)
    output = tr(sample)
    expected = {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[1, 0, 0], [1, 1, 0], [1, 1, 1]]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[1, 0, 3, 2], [0, 1, 2, 3]]
        ),
    }
    assert_matching(output, expected)


def test_random_vertical_flip(sample: Dict[str, Tensor]) -> None:
    tr = transforms.RandomVerticalFlip(p=1)
    output = tr(sample)
    expected = {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[7, 8, 9], [4, 5, 6], [1, 2, 3]]]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[1, 1, 1], [0, 1, 1], [0, 0, 1]]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[0, 1, 2, 3], [1, 0, 3, 2]]
        ),
    }
    assert_matching(output, expected)


def test_identity(sample: Dict[str, Tensor]) -> None:
    tr = transforms.Identity()
    output = tr(sample)
    assert_matching(output, sample)


def test_augmentation_sequential(batch: Dict[str, Tensor]) -> None:
    expected = {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
    }
    augs = transforms.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0),
        data_keys=["image", "mask"],
    )
    output = augs(batch)
    assert_matching(output, expected)
