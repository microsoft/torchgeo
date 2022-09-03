# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict

import kornia.augmentation as K
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torchgeo.transforms import indices, transforms

# Kornia is very particular about its boxes:
#
# * Boxes must have shape B x 4 x 2
# * Defined in clockwise order: top-left, top-right, bottom-right, bottom-left
# * Coordinates must be in (x, y) order
#
# This seems to change with every release...


@pytest.fixture
def batch_gray() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=torch.float),
        "mask": torch.tensor([[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]], dtype=torch.long),
        "boxes": torch.tensor([[[0, 1], [1, 1], [1, 0], [0, 0]]], dtype=torch.float),
        "labels": torch.tensor([[0, 1]]),
    }


@pytest.fixture
def batch_rgb() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(
            [
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                ]
            ],
            dtype=torch.float,
        ),
        "mask": torch.tensor([[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]], dtype=torch.long),
        "boxes": torch.tensor([[[0, 1], [1, 1], [1, 0], [0, 0]]], dtype=torch.float),
        "labels": torch.tensor([[0, 1]]),
    }


@pytest.fixture
def batch_multispectral() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(
            [
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                ]
            ],
            dtype=torch.float,
        ),
        "mask": torch.tensor([[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]], dtype=torch.long),
        "boxes": torch.tensor([[[0, 1], [1, 1], [1, 0], [0, 0]]], dtype=torch.float),
        "labels": torch.tensor([[0, 1]]),
    }


def assert_matching(output: Dict[str, Tensor], expected: Dict[str, Tensor]) -> None:
    for key in expected:
        err = f"output[{key}] != expected[{key}]"
        equal = torch.allclose(output[key], expected[key])
        assert equal, err


def test_augmentation_sequential_gray(batch_gray: Dict[str, Tensor]) -> None:
    expected = {
        "image": torch.tensor([[[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]], dtype=torch.float),
        "mask": torch.tensor([[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=torch.long),
        "boxes": torch.tensor([[[1, 0], [2, 0], [2, 1], [1, 1]]], dtype=torch.float),
        "labels": torch.tensor([[0, 1]]),
    }
    augs = transforms.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0), data_keys=["image", "mask", "boxes"]
    )
    output = augs(batch_gray)
    assert_matching(output, expected)


def test_augmentation_sequential_rgb(batch_rgb: Dict[str, Tensor]) -> None:
    expected = {
        "image": torch.tensor(
            [
                [
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                ]
            ],
            dtype=torch.float,
        ),
        "mask": torch.tensor([[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=torch.long),
        "boxes": torch.tensor([[[1, 0], [2, 0], [2, 1], [1, 1]]], dtype=torch.float),
        "labels": torch.tensor([[0, 1]]),
    }
    augs = transforms.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0), data_keys=["image", "mask", "boxes"]
    )
    output = augs(batch_rgb)
    assert_matching(output, expected)


def test_augmentation_sequential_multispectral(
    batch_multispectral: Dict[str, Tensor]
) -> None:
    expected = {
        "image": torch.tensor(
            [
                [
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                ]
            ],
            dtype=torch.float,
        ),
        "mask": torch.tensor([[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=torch.long),
        "boxes": torch.tensor([[[1, 0], [2, 0], [2, 1], [1, 1]]], dtype=torch.float),
        "labels": torch.tensor([[0, 1]]),
    }
    augs = transforms.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0), data_keys=["image", "mask", "boxes"]
    )
    output = augs(batch_multispectral)
    assert_matching(output, expected)


def test_augmentation_sequential_image_only(
    batch_multispectral: Dict[str, Tensor]
) -> None:
    expected = {
        "image": torch.tensor(
            [
                [
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                ]
            ],
            dtype=torch.float,
        ),
        "mask": torch.tensor([[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]], dtype=torch.long),
        "boxes": torch.tensor([[[0, 1], [1, 1], [1, 0], [0, 0]]], dtype=torch.float),
        "labels": torch.tensor([[0, 1]]),
    }
    augs = transforms.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0), data_keys=["image"]
    )
    output = augs(batch_multispectral)
    assert_matching(output, expected)


def test_sequential_transforms_augmentations(
    batch_multispectral: Dict[str, Tensor]
) -> None:
    expected = {
        "image": torch.tensor(
            [
                [
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                ]
            ],
            dtype=torch.float,
        ),
        "mask": torch.tensor([[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]], dtype=torch.long),
        "boxes": torch.tensor([[[1, 0], [2, 0], [2, 1], [1, 1]]], dtype=torch.float),
        "labels": torch.tensor([[0, 1]]),
    }
    train_transforms = nn.Sequential(
        indices.AppendNBR(index_nir=0, index_swir=0),
        indices.AppendNDBI(index_swir=0, index_nir=0),
        indices.AppendNDSI(index_green=0, index_swir=0),
        indices.AppendNDVI(index_red=0, index_nir=0),
        indices.AppendNDWI(index_green=0, index_nir=0),
        transforms.AugmentationSequential(
            K.RandomHorizontalFlip(p=1.0), data_keys=["image", "mask", "boxes"]
        ),
    )
    output = train_transforms(batch_multispectral)
    assert_matching(output, expected)
