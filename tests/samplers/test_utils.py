# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math

import pytest

from torchgeo.datasets import BoundingBox
from torchgeo.samplers.utils import _to_tuple
from torchgeo.samplers import tile_to_chips

MAYBE_TUPLE = float | tuple[float, float]


@pytest.mark.parametrize(
    "size,stride,expected",
    [
        # size == bounds
        (10, 1, 1),
        (10, None, 1),
        # stride < size
        (8, 1, 3),
        (6, 2, 3),
        (4, 3, 3),
        ((8, 6), (1, 2), (3, 3)),
        ((6, 4), (2, 3), (3, 3)),
        # stride == size
        (3, 3, 4),
        (3, None, 4),
        # stride > size
        (2.5, 3, 4),
    ],
)
def test_tile_to_chips(
    size: MAYBE_TUPLE, stride: MAYBE_TUPLE | None, expected: MAYBE_TUPLE
) -> None:
    bounds = BoundingBox(0, 10, 20, 30, 40, 50)
    size = _to_tuple(size)
    if stride is not None:
        stride = _to_tuple(stride)
    expected = _to_tuple(expected)
    rows, cols = tile_to_chips(bounds, size, stride)
    assert math.isclose(rows, expected[0])
    assert math.isclose(cols, expected[1])
