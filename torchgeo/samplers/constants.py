# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler constants."""

from enum import Enum, auto


class Units(Enum):
    """Enumeration defining units of ``size`` parameter.

    Used by :class:`~torchgeo.sampler.GeoSampler` and
    :class:`~torchgeo.sampler.BatchGeoSampler`.
    """

    PIXELS = auto()
    CRS = auto()
