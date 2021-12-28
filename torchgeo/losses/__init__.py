# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo losses."""

from .qr import QRLoss, RQLoss

__all__ = ("QRLoss", "RQLoss")

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.losses"
