# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo losses."""

from .qr import QRLoss, RQLoss
from .ssl import NTXentLoss

__all__ = ("NTXentLoss", "QRLoss", "RQLoss")
