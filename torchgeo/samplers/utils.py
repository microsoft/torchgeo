# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler utilities."""

import random
from typing import Tuple, Union

from torchgeo.datasets.utils import BoundingBox

from .constants import SIZE_IN_PIXELS, SIZE_IN_CRS_UNITS


def _to_tuple(value: Union[Tuple[float, float], float]) -> Tuple[float, float]:
    """Convert value to a tuple if it is not already a tuple.

    Args:
        value: input value

    Returns:
        value if value is a tuple, else (value, value)
    """
    if isinstance(value, (float, int)):
        return (value, value)
    else:
        return value


def get_random_bounding_box(
    bounds: BoundingBox,
    size: Union[Tuple[float, float], float],
    res: float,
    sample_mode: int,
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
        sample_mode: whether to sample in pixel space or CRS unit space

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    t_size: Tuple[float, float] = _to_tuple(size)

    width = (bounds.maxx - bounds.minx - t_size[1]) // res
    minx = random.randrange(int(width)) * res + bounds.minx
    if sample_mode == SIZE_IN_CRS_UNITS:
        maxx = minx + t_size[1]
    elif sample_mode == SIZE_IN_PIXELS:
        maxx = minx + t_size[1] * res

    height = (bounds.maxy - bounds.miny - t_size[0]) // res
    miny = random.randrange(int(height)) * res + bounds.miny
    if sample_mode == SIZE_IN_CRS_UNITS:
        maxy = miny + t_size[0]
    elif sample_mode == SIZE_IN_PIXELS:
        maxy = miny + t_size[0] * res

    mint = bounds.mint
    maxt = bounds.maxt

    query = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
    return query
