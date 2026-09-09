# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for branch coverage of moco_augmentations in trainers/moco.py.

Relates to: https://github.com/microsoft/torchgeo/pull/3549
"""

import torch

from torchgeo.trainers.moco import moco_augmentations


def test_moco_augmentations_v1() -> None:
    """Test moco_augmentations with version 1."""
    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    aug1, aug2 = moco_augmentations(version=1, size=64, weights=weights)
    assert aug1 is aug2  # version 1 uses same augmentation


def test_moco_augmentations_v2() -> None:
    """Test moco_augmentations with version 2."""
    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    aug1, aug2 = moco_augmentations(version=2, size=64, weights=weights)
    assert aug1 is aug2  # version 2 uses same augmentation


def test_moco_augmentations_v3() -> None:
    """Test moco_augmentations with version 3 (else branch)."""
    weights = torch.tensor([0.2126, 0.7152, 0.0722])
    aug1, aug2 = moco_augmentations(version=3, size=64, weights=weights)
    assert aug1 is not aug2  # version 3 uses different augmentations
