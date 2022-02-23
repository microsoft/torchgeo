# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler constants."""

from enum import Enum, auto


class Units(Enum):
    """Enumeration defining units of ``size`` parameter.

    Used by :class:`~torchgeo.samplers.GeoSampler` and
    :class:`~torchgeo.samplers.BatchGeoSampler`.
    """

    #: Units in number of pixels
    PIXELS = auto()

    #: Units of :term:`coordinate reference system (CRS)`
    CRS = auto()
