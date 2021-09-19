# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo models."""

from .fccd import FCEF, FCSiamConc, FCSiamDiff
from .fcn import FCN
from .farseg import FarSeg

__all__ = ("FCN", "FCEF", "FCSiamConc", "FCSiamDiff", "FarSeg")

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.models"
