# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo losses."""

from .focaljaccard import BinaryFocalJaccardLoss
from .qr import QRLoss, RQLoss
from .xentjaccard import BinaryXEntJaccardLoss

__all__ = ('QRLoss', 'RQLoss', 'BinaryFocalJaccardLoss', 'BinaryXEntJaccardLoss')
