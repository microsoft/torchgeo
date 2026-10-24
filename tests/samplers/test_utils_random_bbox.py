# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for branch coverage of get_random_bounding_box in samplers/utils.py.

Relates to: https://github.com/microsoft/torchgeo/pull/3549
"""

import torch

from torchgeo.samplers.utils import get_random_bounding_box


def test_get_random_bounding_box_with_float_size() -> None:
    """Test with single float for size and res."""
    bounds = (0.0, 0.0, 10.0, 10.0)
    slices = get_random_bounding_box(bounds, size=2.0, res=1.0)
    assert len(slices) == 2
    x_slice, y_slice = slices
    assert x_slice.stop - x_slice.start == 2.0
    assert y_slice.stop - y_slice.start == 2.0


def test_get_random_bounding_box_with_tuple_size() -> None:
    """Test with tuple for size and res."""
    bounds = (0.0, 0.0, 10.0, 10.0)
    slices = get_random_bounding_box(bounds, size=(2.0, 3.0), res=(1.0, 1.0))
    assert len(slices) == 2
    x_slice, y_slice = slices
    assert x_slice.stop - x_slice.start == 3.0
    assert y_slice.stop - y_slice.start == 2.0


def test_get_random_bounding_box_with_generator() -> None:
    """Test with explicit generator for reproducibility."""
    bounds = (0.0, 0.0, 100.0, 100.0)
    gen = torch.Generator()
    gen.manual_seed(42)
    slices1 = get_random_bounding_box(bounds, size=10.0, res=1.0, generator=gen)

    gen.manual_seed(42)
    slices2 = get_random_bounding_box(bounds, size=10.0, res=1.0, generator=gen)
    assert slices1[0].start == slices2[0].start
    assert slices1[1].start == slices2[1].start


def test_get_random_bounding_box_without_generator() -> None:
    """Test without generator (default None branch)."""
    bounds = (0.0, 0.0, 20.0, 20.0)
    slices = get_random_bounding_box(bounds, size=5.0, res=1.0)
    assert len(slices) == 2
