# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo models."""

from .changestar import ChangeMixin, ChangeStar, ChangeStarFarSeg
from .farseg import FarSeg
from .fcn import FCN
from .fcsiam import FCSiamConc, FCSiamDiff
from .rcf import RCF
from .resnet import resnet50

__all__ = (
    "ChangeMixin",
    "ChangeStar",
    "ChangeStarFarSeg",
    "FarSeg",
    "FCN",
    "FCSiamConc",
    "FCSiamDiff",
    "RCF",
    "resnet50",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.models"
