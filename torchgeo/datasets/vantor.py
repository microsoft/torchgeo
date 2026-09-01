# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""
Vantor dataset (formerly known as Vantor)
"""

from .geo import RasterDataset


class Vantor(RasterDataset):
    """
    Vantor dataset (formerly known as Maxar)
    """

    filename_glob = '*.tif'
