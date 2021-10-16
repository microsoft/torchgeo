# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from .indices import AppendNDBI, AppendNDSI, AppendNDVI, AppendNDWI
from .transforms import AugmentationSequential

__all__ = (
    "AppendNDBI",
    "AppendNDSI",
    "AppendNDVI",
    "AppendNDWI",
    "AugmentationSequential",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.transforms"
