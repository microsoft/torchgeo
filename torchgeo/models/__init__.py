# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo models."""

from .api import get_model, get_model_weights, get_weight, list_models
from .changestar import ChangeMixin, ChangeStar, ChangeStarFarSeg
from .croma import CROMA, CROMABase_Weights, CROMALarge_Weights, croma_base, croma_large
from .dofa import (
    DOFA,
    DOFABase16_Weights,
    DOFALarge16_Weights,
    dofa_base_patch16_224,
    dofa_huge_patch14_224,
    dofa_large_patch16_224,
    dofa_small_patch16_224,
)
from .farseg import FarSeg
from .fcn import FCN
from .fcsiam import FCSiamConc, FCSiamDiff
from .rcf import RCF
from .resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    resnet18,
    resnet50,
    resnet152,
)
from .scale_mae import ScaleMAE, ScaleMAELarge16_Weights, scalemae_large_patch16
from .swin import Swin_V2_B_Weights, Swin_V2_T_Weights, swin_v2_b, swin_v2_t
from .vit import (
    ViTBase16_Weights,
    ViTHuge14_Weights,
    ViTLarge16_Weights,
    ViTSmall16_Weights,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch16_224,
)

__all__ = (
    'CROMA',
    'DOFA',
    'FCN',
    'RCF',
    'CROMABase_Weights',
    'CROMALarge_Weights',
    'ChangeMixin',
    'ChangeStar',
    'ChangeStarFarSeg',
    'DOFABase16_Weights',
    'DOFALarge16_Weights',
    'FCSiamConc',
    'FCSiamDiff',
    'FarSeg',
    'ResNet18_Weights',
    'ResNet50_Weights',
    'ResNet152_Weights',
    'ScaleMAE',
    'ScaleMAELarge16_Weights',
    'Swin_V2_B_Weights',
    'Swin_V2_T_Weights',
    'ViTBase16_Weights',
    'ViTHuge14_Weights',
    'ViTLarge16_Weights',
    'ViTSmall16_Weights',
    'croma_base',
    'croma_large',
    'dofa_base_patch16_224',
    'dofa_huge_patch14_224',
    'dofa_large_patch16_224',
    'dofa_small_patch16_224',
    'get_model',
    'get_model_weights',
    'get_weight',
    'list_models',
    'resnet18',
    'resnet50',
    'resnet152',
    'scalemae_large_patch16',
    'swin_v2_b',
    'swin_v2_t',
    'vit_base_patch16_224',
    'vit_huge_patch14_224',
    'vit_large_patch16_224',
    'vit_small_patch16_224',
)
