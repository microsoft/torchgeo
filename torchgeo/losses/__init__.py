# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo losses."""

from .elects import EarlyRewardLoss
from .qr import QRLoss, RQLoss
from .quantile import PinballLoss

__all__ = ('EarlyRewardLoss', 'PinballLoss', 'QRLoss', 'RQLoss')
