# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler utilities."""

import random
from typing import Tuple, Union

from torchgeo.datasets.utils import BoundingBox


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
    bounds: BoundingBox, size: Union[Tuple[float, float], float]
) -> BoundingBox:
    """Returns a random bounding box within a given bounding box.

    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    t_size: Tuple[float, float] = _to_tuple(size)

    minx = random.uniform(bounds.minx, bounds.maxx - t_size[1])
    maxx = minx + t_size[1]

    miny = random.uniform(bounds.miny, bounds.maxy - t_size[0])
    maxy = miny + t_size[0]

    mint = bounds.mint
    maxt = bounds.maxt

    return BoundingBox(minx, maxx, miny, maxy, mint, maxt)
