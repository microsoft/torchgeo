# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Metric utilities."""

import torch
from torch import Tensor


def latitude_weights(latitude: Tensor) -> Tensor:
    r"""Compute latitude/area weights.

    .. math::

       w(i) = \frac{
           \sin \theta_i^u - \sin \theta_i^l
       }{
           \frac{1}{I} \sum_i^I \left(\sin \theta_i^u - \sin \theta_i^l\right)
       }

    where :math:`\theta_i^u` and :math:`\theta_i^l` indicate upper and lower
    latitude bounds, respectively.

    For more details on this formulation, see equation 1 in:

    * https://arxiv.org/abs/2308.15560

    Args:
        latitude: Latitude of :math:`f` and :math:`o` in degrees
            :math:`[-90, \ldots, 90]`.

    Returns:
        Relative area of each grid cell.

    Raises:
        AssertionError: If *latitude* is not in ascending order.

    .. versionadded:: 0.11
    """
    # Reference implementation:
    # https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py

    assert (torch.diff(latitude) > 0).all()

    theta = torch.deg2rad(latitude)
    pi_over_2 = torch.tensor([torch.pi / 2])
    bounds = torch.cat([-pi_over_2, (theta[:-1] + theta[1:]) / 2, pi_over_2])
    upper = bounds[1:]
    lower = bounds[:-1]
    weights = torch.sin(upper) - torch.sin(lower)
    weights /= torch.mean(weights)
    return weights
