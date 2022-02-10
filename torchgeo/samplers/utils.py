# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler utilities."""

import random
from typing import Tuple, Union

from ..datasets import BoundingBox

from .constants import Units


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
    units: Union[Units, int],
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
        units: defines if `size` is in pixels or in CRS units.

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    t_size: Tuple[float, float] = _to_tuple(size)
    if units == Units.PIXELS:
        # We have to re-assign t_size because tuples are immutable
        t_size = (t_size[0] * res, t_size[1] * res)

    width = (bounds.maxx - bounds.minx - t_size[1]) // res
    minx = random.randrange(int(width)) * res + bounds.minx
    maxx = minx + t_size[1]

    height = (bounds.maxy - bounds.miny - t_size[0]) // res
    miny = random.randrange(int(height)) * res + bounds.miny
    maxy = miny + t_size[0]

    mint = bounds.mint
    maxt = bounds.maxt

    query = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
    return query
