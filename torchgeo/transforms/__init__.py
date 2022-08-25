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
    "AppendBNDVI",
    "AppendGBNDVI",
    "AppendGNDVI",
    "AppendGRNDVI",
    "AppendNBR",
    "AppendNDBI",
    "AppendNDRE",
    "AppendNDSI",
    "AppendNDVI",
    "AppendNDWI",
    "AppendNormalizedDifferenceIndex",
    "AppendRBNDVI",
    "AppendSWI",
    "AppendTriBandNormalizedDifferenceIndex",
    "AugmentationSequential",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.transforms"
