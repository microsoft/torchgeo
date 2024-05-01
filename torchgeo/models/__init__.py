# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo models."""

from .api import get_model, get_model_weights, get_weight, list_models
from .changestar import ChangeMixin, ChangeStar, ChangeStarFarSeg
from .dofa import (
    DOFA,
    DOFABase16_Weights,
    DOFALarge16_Weights,
    dofa_base_patch16_224,
    dofa_huge_patch16_224,
    dofa_large_patch16_224,
    dofa_small_patch16_224,
)
from .farseg import FarSeg
from .fcn import FCN
from .fcsiam import FCSiamConc, FCSiamDiff
from .rcf import RCF
from .resnet import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from .swin import Swin_V2_B_Weights, swin_v2_b
from .vit import ViTSmall16_Weights, vit_small_patch16_224

__all__ = (
    # models
    "ChangeMixin",
    "ChangeStar",
    "ChangeStarFarSeg",
    "DOFA",
    "dofa_small_patch16_224",
    "dofa_base_patch16_224",
    "dofa_large_patch16_224",
    "dofa_huge_patch16_224",
    "FarSeg",
    "FCN",
    "FCSiamConc",
    "FCSiamDiff",
    "RCF",
    "resnet18",
    "resnet50",
    "swin_v2_b",
    "vit_small_patch16_224",
    # weights
    "DOFABase16_Weights",
    "DOFALarge16_Weights",
    "ResNet50_Weights",
    "ResNet18_Weights",
    "Swin_V2_B_Weights",
    "ViTSmall16_Weights",
    # utilities
    "get_model",
    "get_model_weights",
    "get_weight",
    "list_models",
)
