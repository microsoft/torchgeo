# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Common sampler utilities."""

import math
from typing import overload

import numpy as np
import torch
from numpy.typing import NDArray
from pandas import Timedelta
from torch import Generator
from typing_extensions import deprecated


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
    if isinstance(value, int | float):
        return (value, value)
    else:
        return value


@deprecated('Use geopandas.GeoSeries.sample_points instead')
def get_random_bounding_box(
    bounds: tuple[float, float, float, float],
    size: tuple[float, float] | float,
    res: tuple[float, float] | float,
    generator: Generator | None = None,
) -> tuple[slice, slice]:
    """Returns a random bounding box within a given bounding box.

    The ``size`` argument can either be:

        * a single ``float`` - in which case the same value is used for the height and
          width dimension
        * a ``tuple`` of two floats - in which case, the first *float* is used for the
          height dimension, and the second *float* for the width dimension

    .. versionadded:: 0.7
        The *generator* parameter.

    Args:
        bounds: the larger bounding box to sample from
        size: the size of the bounding box to sample
        res: the resolution of the image
        generator: pseudo-random number generator (PRNG).

    Returns:
        randomly sampled bounding box from the extent of the input
    """
    xmin, ymin, xmax, ymax = bounds
    t_size = _to_tuple(size)
    t_res = _to_tuple(res)

    # May be negative if bounding box is smaller than patch size
    width = (xmax - xmin - t_size[1]) / t_res[0]
    height = (ymax - ymin - t_size[0]) / t_res[1]

    # Use an integer multiple of res to avoid resampling
    xmin += int(torch.rand(1, generator=generator).item() * width) * t_res[0]
    ymin += int(torch.rand(1, generator=generator).item() * height) * t_res[1]

    xmax = xmin + t_size[1]
    ymax = ymin + t_size[0]

    return slice(xmin, xmax), slice(ymin, ymax)


@deprecated('Use torchgeo.samplers.utils.convolution_arithmetic instead')
def tile_to_chips(
    bounds: tuple[float, float, float, float],
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

    xmin, ymin, xmax, ymax = bounds

    rows = math.ceil((ymax - ymin - size[0]) / stride[0]) + 1
    cols = math.ceil((xmax - xmin - size[1]) / stride[1]) + 1

    return rows, cols


def convolution_arithmetic[T: (float, Timedelta)](
    input_size: T, kernel_size: T, stride: T | None = None
) -> int:
    r"""Compute number of spatial/temporal windows that can be sampled via convolution.

    Let :math:`i` be the size of the input window.
    Let :math:`k` be the requested size of the output window.
    Let :math:`s` be the requested stride.
    Let :math:`o` be the number of output windows sampled from each input.

    :math:`o` can then be computed as:

    .. math::

       o = \left\lceil \frac{i - k}{s} \right\rceil + 1

    This is almost identical to relationship 5 in
    https://doi.org/10.48550/arXiv.1603.07285. However, we use ceiling instead of floor
    because we want to include the final remaining window in each input when
    *input_size* is not an integer multiple of *stride*.

    Args:
        input_size: Size of the input window.
        kernel_size: Size of each output window.
        stride: Stride with which to sample (defaults to *input_size*).

    Returns:
        The number of output windows that can be sampled.

    .. versionadded:: 0.10
    """
    stride = stride or kernel_size
    return math.ceil((input_size - kernel_size) / stride) + 1


def prism(x: NDArray, y: NDArray, z: NDArray) -> list[NDArray]:
    """Convert x, y, z coordinates to the vertices of a prism.

    Args:
        x: All x coordinates of a Polygon.
        y: All y coordinates of a Polygon.
        z: Two z coordinates to project the Polygon into.

    Returns:
        The vertices of a 3D prism.

    Raises:
        AssertionError: If len(x) != len(y) or len(z) != 2.
    """
    assert len(x) == len(y)
    assert len(z) == 2
    verts = []

    # Bottom face
    z0 = z[0].repeat(len(x))
    verts.append(np.stack([x, y, z0]).T)

    # Top face
    z1 = z[1].repeat(len(x))
    verts.append(np.stack([x, y, z1]).T)

    # Side faces
    zi = np.array([z[0], z[0], z[1], z[1], z[0]])
    for i in range(len(x) - 1):
        xi = np.array([x[i], x[i + 1], x[i + 1], x[i], x[i]])
        yi = np.array([y[i], y[i + 1], y[i + 1], y[i], y[i]])
        verts.append(np.stack([xi, yi, zi]).T)

    return verts
