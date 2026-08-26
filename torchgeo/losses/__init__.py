# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo losses."""

from .elects import EarlyRewardLoss
from .qr import QRLoss, RQLoss

__all__ = ('EarlyRewardLoss', 'QRLoss', 'RQLoss')
