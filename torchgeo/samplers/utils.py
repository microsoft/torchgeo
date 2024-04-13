# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler utilities."""

import math
from typing import overload

import torch

from ..datasets import BoundingBox


@overload
def _to_tuple(value: tuple[int, int] | int) -> tuple[int, int]: ...


@overload
def _to_tuple(value: tuple[float, float] | float) -> tuple[float, float]: ...


def _to_tuple(value: tuple[float, float] | float) -> tuple[float, float]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, float | int):
        return (value, value)
    else:
        return value


def get_random_bounding_box(
    bounds: BoundingBox, size: tuple[float, float] | float, res: float
) -> BoundingBox:
    """Returns a random bounding box within a given bounding box.

    The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample
        res: the resolution of the image

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    t_size = _to_tuple(size)

    width = (bounds.maxx - bounds.minx - t_size[1]) // res
    height = (bounds.maxy - bounds.miny - t_size[0]) // res

    minx = bounds.minx
    miny = bounds.miny

    # random.randrange crashes for inputs <= 0
    if width > 0:
        minx += torch.rand(1).item() * width * res
    if height > 0:
        miny += torch.rand(1).item() * height * res

    maxx = minx + t_size[1]
    maxy = miny + t_size[0]

    mint = bounds.mint
    maxt = bounds.maxt

    query = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
    return query


def tile_to_chips(
    bounds: BoundingBox,
    size: tuple[float, float],
    stride: tuple[float, float] | None = None,
) -> tuple[int, int]:
    r"""Compute number of :term:`chips <chip>` that can be sampled from a :term:`tile`.

    Let :math:`i` be the size of the input tile. Let :math:`k` be the requested size of
    the output patch. Let :math:`s` be the requested stride. Let :math:`o` be the number
    of output chips sampled from each tile. :math:`o` can then be computed as:

    .. math::

       o = \left\lceil \frac{i - k}{s} \right\rceil + 1

    This is almost identical to relationship 5 in
    https://doi.org/10.48550/arXiv.1603.07285. However, we use ceiling instead of floor
    because we want to include the final remaining chip in each row/column when bounds
    is not an integer multiple of stride.

    Args:
        bounds: bounding box of tile
        size: size of output patch
        stride: stride with which to sample (defaults to ``size``)

    Returns:
        the number of rows/columns that can be sampled

    .. versionadded:: 0.4
    """
    if stride is None:
        stride = size

    assert stride[0] > 0
    assert stride[1] > 0

    rows = math.ceil((bounds.maxy - bounds.miny - size[0]) / stride[0]) + 1
    cols = math.ceil((bounds.maxx - bounds.minx - size[1]) / stride[1]) + 1

    return rows, cols
