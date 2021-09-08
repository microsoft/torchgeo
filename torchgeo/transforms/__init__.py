# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from .transforms import (
    CombinePatches,
    ExtractPatches,
    Identity,
    RandomHorizontalFlip,
    RandomVerticalFlip,
)

__all__ = (
    "Identity",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "ExtractPatches",
    "CombinePatches",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.transforms"
