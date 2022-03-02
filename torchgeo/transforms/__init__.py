# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from .indices import (
    AppendBNDVI,
    AppendGBNDVI,
    AppendGNDVI,
    AppendGRNDVI,
    AppendNBR,
    AppendNDBI,
    AppendNDRE,
    AppendNDSI,
    AppendNDVI,
    AppendNDWI,
    AppendNormalizedDifferenceIndex,
    AppendRBNDVI,
    AppendSWI,
    AppendTriBandNormalizedDifferenceIndex,
)
from .transforms import AugmentationSequential

__all__ = (
    "AppendNormalizedDifferenceIndex",
    "AppendGBNDVI",
    "AppendBNDVI",
    "AppendGNDVI",
    "AppendGBNDVI",
    "AppendNBR",
    "AppendNDBI",
    "AppendNDRE",
    "AppendNDSI",
    "AppendNDVI",
    "AppendNDWI",
    "AppendRBNDVI",
    "AppendSWI",
    "AugmentationSequential",
    "AppendTriBandNormalizedDifferenceIndex",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.transforms"
