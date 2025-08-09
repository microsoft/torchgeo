# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo models."""

from .api import get_model, get_model_weights, get_weight, list_models
from .changestar import ChangeMixin, ChangeStar, ChangeStarFarSeg
from .changevit import changevit_small, changevit_tiny
from .copernicusfm import CopernicusFM, CopernicusFM_Base_Weights, copernicusfm_base
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
from .earthloc import EarthLoc, EarthLoc_Weights, earthloc
from .farseg import FarSeg
from .fcn import FCN
from .fcsiam import FCSiamConc, FCSiamDiff
from .ltae import LTAE
from .panopticon import Panopticon, Panopticon_Weights, panopticon_vitb14
from .rcf import MOSAIKS, RCF
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
from .unet import Unet_Weights, unet
from .vit import (
    ViTBase14_DINOv2_Weights,
    ViTBase16_Weights,
    ViTHuge14_Weights,
    ViTLarge16_Weights,
    ViTSmall14_DINOv2_Weights,
    ViTSmall16_Weights,
    vit_base_patch14_dinov2,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch14_dinov2,
    vit_small_patch16_224,
)
from .yolo import YOLO_Weights, yolo

__all__ = (
    'CROMA',
    'DOFA',
    'FCN',
    'LTAE',
    'MOSAIKS',
    'RCF',
    'CROMABase_Weights',
    'CROMALarge_Weights',
    'ChangeMixin',
    'ChangeStar',
    'ChangeStarFarSeg',
    'CopernicusFM',
    'CopernicusFM_Base_Weights',
    'DOFABase16_Weights',
    'DOFALarge16_Weights',
    'EarthLoc',
    'EarthLoc_Weights',
    'FCSiamConc',
    'FCSiamDiff',
    'FarSeg',
    'Panopticon',
    'Panopticon_Weights',
    'ResNet18_Weights',
    'ResNet50_Weights',
    'ResNet152_Weights',
    'ScaleMAE',
    'ScaleMAELarge16_Weights',
    'Swin_V2_B_Weights',
    'Swin_V2_T_Weights',
    'Unet_Weights',
    'ViTBase14_DINOv2_Weights',
    'ViTBase16_Weights',
    'ViTHuge14_Weights',
    'ViTLarge16_Weights',
    'ViTSmall14_DINOv2_Weights',
    'ViTSmall16_Weights',
    'YOLO_Weights',
    'changevit_small',
    'changevit_tiny',
    'copernicusfm_base',
    'croma_base',
    'croma_large',
    'dofa_base_patch16_224',
    'dofa_huge_patch14_224',
    'dofa_large_patch16_224',
    'dofa_small_patch16_224',
    'earthloc',
    'get_model',
    'get_model_weights',
    'get_weight',
    'list_models',
    'panopticon_vitb14',
    'resnet18',
    'resnet50',
    'resnet152',
    'scalemae_large_patch16',
    'swin_v2_b',
    'swin_v2_t',
    'unet',
    'vit_base_patch14_dinov2',
    'vit_base_patch16_224',
    'vit_huge_patch14_224',
    'vit_large_patch16_224',
    'vit_small_patch14_dinov2',
    'vit_small_patch16_224',
    'yolo',
)
