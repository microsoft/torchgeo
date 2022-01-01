# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from .indices import (
    AppendNBR,
    AppendNDBI,
    AppendNDSI,
    AppendNDVI,
    AppendNDWI,
    AppendNormalizedDifferenceIndex,
)
from .transforms import AugmentationSequential

__all__ = (
    "AppendNormalizedDifferenceIndex",
    "AppendNBR",
    "AppendNDBI",
    "AppendNDSI",
    "AppendNDVI",
    "AppendNDWI",
    "AugmentationSequential",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.transforms"
