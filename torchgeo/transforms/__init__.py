# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from .indices import (
    AppendBNDVI,
    AppendGNDVI,
    AppendNBR,
    AppendNDBI,
    AppendNDRE,
    AppendNDSI,
    AppendNDVI,
    AppendNDWI,
    AppendNormalizedDifferenceIndex,
    AppendSWI,
    AppendTriBandNormalizedDifferenceIndex,
)
from .transforms import AugmentationSequential

__all__ = (
    "AppendNormalizedDifferenceIndex",
    "AppendBNDVI",
    "AppendGNDVI",
    "AppendNBR",
    "AppendNDBI",
    "AppendNDRE",
    "AppendNDSI",
    "AppendNDVI",
    "AppendNDWI",
    "AppendSWI",
    "AugmentationSequential",
    "AppendTriBandNormalizedDifferenceIndex",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.transforms"
