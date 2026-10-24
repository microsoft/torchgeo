# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for branch coverage of collate_fn_detection in datamodules/utils.py.

Relates to: https://github.com/microsoft/torchgeo/pull/3549
"""

from typing import Any

import torch

from torchgeo.datamodules.utils import collate_fn_detection
from torchgeo.datasets.utils import Sample


def _make_sample(**kwargs: Any) -> Sample:
    """Create a sample dict with image and optional bbox/label/mask."""
    sample: Sample = {'image': torch.zeros(3, 4, 4)}
    sample.update(kwargs)
    return sample


def test_collate_fn_detection_image_only() -> None:
    """Test with only image - no bbox_xyxy, label, or mask branches."""
    batch = [_make_sample(), _make_sample()]
    result = collate_fn_detection(batch)
    assert 'image' in result
    assert result['image'].shape == (2, 3, 4, 4)
    assert 'bbox_xyxy' not in result
    assert 'label' not in result
    assert 'mask' not in result


def test_collate_fn_detection_with_bbox_and_label() -> None:
    """Test with bbox_xyxy and label - exercises those branches."""
    batch = [
        _make_sample(
            bbox_xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]]), label=torch.tensor([1])
        ),
        _make_sample(
            bbox_xyxy=torch.tensor([[1.0, 1.0, 3.0, 3.0]]), label=torch.tensor([2])
        ),
    ]
    result = collate_fn_detection(batch)
    assert 'bbox_xyxy' in result
    assert 'label' in result
    assert len(result['bbox_xyxy']) == 2
    assert len(result['label']) == 2


def test_collate_fn_detection_bbox_without_label() -> None:
    """Test with bbox_xyxy but no label - exercises elif branch for auto-label."""
    batch = [
        _make_sample(bbox_xyxy=torch.tensor([[0.0, 0.0, 2.0, 2.0]])),
        _make_sample(bbox_xyxy=torch.tensor([[1.0, 1.0, 3.0, 3.0]])),
    ]
    result = collate_fn_detection(batch)
    assert 'bbox_xyxy' in result
    assert 'label' in result
    # Auto-label should be all 1s
    assert all(torch.all(lbl == 1) for lbl in result['label'])


def test_collate_fn_detection_with_mask() -> None:
    """Test with mask - exercises mask branch."""
    batch = [
        _make_sample(mask=torch.zeros(1, 4, 4, dtype=torch.long)),
        _make_sample(mask=torch.ones(1, 4, 4, dtype=torch.long)),
    ]
    result = collate_fn_detection(batch)
    assert 'mask' in result
    assert len(result['mask']) == 2
