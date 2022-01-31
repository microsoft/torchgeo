# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Dict

import kornia.augmentation as K
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from torchgeo.transforms import indices, transforms

# Tests require newer version of Kornia for newer bounding box behavior
pytest.importorskip("kornia", minversion="0.6.3")


@pytest.fixture
def batch_gray() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        # This is a list of 4 (y,x) points of the corners of a bounding box.
        # kornia expects something with (B, 4, 2) shape
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[[0, 0], [0, 1], [1, 1], [1, 0]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
    }


@pytest.fixture
def batch_rgb() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                ]
            ],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[[0, 0], [0, 1], [1, 1], [1, 0]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
    }


@pytest.fixture
def batch_multispectral() -> Dict[str, Tensor]:
    return {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [
                [
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                    [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                ]
            ],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[[0, 0], [0, 1], [1, 1], [1, 0]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
    }


def assert_matching(output: Dict[str, Tensor], expected: Dict[str, Tensor]) -> None:
    for key in expected:
        err = f"output[{key}] != expected[{key}]"
        equal = torch.allclose(output[key], expected[key])  # type: ignore[attr-defined]
        assert equal, err


def test_augmentation_sequential_gray(batch_gray: Dict[str, Tensor]) -> None:
    expected = {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [[[[3, 2, 1], [6, 5, 4], [9, 8, 7]]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[[1, 0], [2, 0], [2, 1], [1, 1]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
    }
    augs = transforms.AugmentationSequential(
        K.RandomHorizontalFlip(p=1.0), data_keys=["image", "mask", "boxes"]
    )
    output = augs(batch_gray)
    assert_matching(output, expected)


def test_augmentation_sequential_rgb(batch_rgb: Dict[str, Tensor]) -> None:
    expected = {
        "image": torch.tensor(  # type: ignore[attr-defined]
            [
                [
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                ]
            ],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[[1, 0], [2, 0], [2, 1], [1, 1]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
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
        "image": torch.tensor(  # type: ignore[attr-defined]
            [
                [
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                ]
            ],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[[1, 0], [2, 0], [2, 1], [1, 1]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
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
        "image": torch.tensor(  # type: ignore[attr-defined]
            [
                [
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                    [[3, 2, 1], [6, 5, 4], [9, 8, 7]],
                ]
            ],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[0, 0, 1], [0, 1, 1], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[[0, 0], [0, 1], [1, 1], [1, 0]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
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
        "image": torch.tensor(  # type: ignore[attr-defined]
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
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "mask": torch.tensor(  # type: ignore[attr-defined]
            [[[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]],
            dtype=torch.long,  # type: ignore[attr-defined]
        ),
        "boxes": torch.tensor(  # type: ignore[attr-defined]
            [[[1, 0], [2, 0], [2, 1], [1, 1]]],
            dtype=torch.float,  # type: ignore[attr-defined]
        ),
        "labels": torch.tensor([[0, 1]]),  # type: ignore[attr-defined]
    }
    train_transforms = nn.Sequential(  # type: ignore[attr-defined]
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
