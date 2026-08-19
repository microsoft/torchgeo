# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import math

import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pandas import Timedelta

from torchgeo.samplers import get_random_bounding_box, tile_to_chips
from torchgeo.samplers.utils import _to_tuple, convolution_arithmetic, prism

MAYBE_TUPLE = float | tuple[float, float]


@pytest.mark.parametrize(
    'value,expected',
    [(5, (5, 5)), (3.14, (3.14, 3.14)), ((4, 8), (4, 8)), ((2.5, 7.5), (2.5, 7.5))],
)
def test_to_tuple(value: MAYBE_TUPLE, expected: tuple[float, float]) -> None:
    assert _to_tuple(value) == expected


@pytest.mark.parametrize(
    'size,res',
    [
        # size == bounds
        (10, 1),
        (5, 2),
        ((10, 10), 1),
        ((5, 5), 2),
        # size < bounds
        (5, 1),
        (3.14, 1),
        ((2.5, 7.5), 1),
    ],
)
def test_get_random_bounding_box(size: MAYBE_TUPLE, res: MAYBE_TUPLE) -> None:
    bounds = (0, 20, 10, 30)
    with pytest.deprecated_call(match='Use .* instead'):
        x, y = get_random_bounding_box(bounds, size, res)
    assert bounds[0] <= x.start < x.stop <= bounds[2]
    assert bounds[1] <= y.start < y.stop <= bounds[3]


@pytest.mark.parametrize(
    'size,stride,expected',
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
    bounds = (0, 20, 10, 30)
    size = _to_tuple(size)
    if stride is not None:
        stride = _to_tuple(stride)
    expected = _to_tuple(expected)
    with pytest.deprecated_call(match='Use .* instead'):
        rows, cols = tile_to_chips(bounds, size, stride)
    assert math.isclose(rows, expected[0])
    assert math.isclose(cols, expected[1])


@pytest.mark.parametrize(
    'input_size,kernel_size,stride,expected',
    [
        # i == k
        (10, 10, 1, 1),
        (Timedelta(days=10), Timedelta(days=10), Timedelta(days=1), 1),
        # s < k
        (10, 2, 1, 9),
        (Timedelta(hours=10), Timedelta(hours=2), Timedelta(hours=1), 9),
        # s == k
        (10, 2, 2, 5),
        (10, 2, None, 5),
        (Timedelta(minutes=10), Timedelta(minutes=2), Timedelta(minutes=2), 5),
        (Timedelta(minutes=10), Timedelta(minutes=2), None, 5),
        # s > k
        (10, 2, 3, 4),
        (Timedelta(seconds=10), Timedelta(seconds=2), Timedelta(seconds=3), 4),
    ],
)
def test_convolution_arithmetic[T: (float, Timedelta)](
    input_size: T, kernel_size: T, stride: T | None, expected: int
) -> None:
    output_size = convolution_arithmetic(input_size, kernel_size, stride)
    assert output_size == expected


def test_prism() -> None:
    x = np.array([1, 2, 1, 0, 1])
    y = np.array([0, 1, 2, 1, 0])
    z = np.array([0, 1])
    verts = prism(x, y, z)
    Poly3DCollection(verts)
