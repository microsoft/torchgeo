# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common sampler constants."""

from enum import Enum


class Units(Enum):
    """Enumeration to define units of `size` used for GeoSampler."""

    PIXELS = 0
    CRS = 1
