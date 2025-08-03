# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""APIs for querying and loading pre-trained model weights.

See the following references for design details:

* https://pytorch.org/blog/easily-list-and-initialize-models-with-new-apis-in-torchvision/
* https://pytorch.org/vision/stable/models.html
* https://github.com/pytorch/vision/blob/main/torchvision/models/_api.py
"""

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torchvision.models._api import WeightsEnum

from .copernicusfm import CopernicusFM_Base_Weights, copernicusfm_base
from .croma import CROMABase_Weights, CROMALarge_Weights, croma_base, croma_large
from .dofa import (
    DOFABase16_Weights,
    DOFALarge16_Weights,
    dofa_base_patch16_224,
    dofa_huge_patch14_224,
    dofa_large_patch16_224,
    dofa_small_patch16_224,
)
from .earthloc import EarthLoc_Weights, earthloc
from .panopticon import Panopticon_Weights, panopticon_vitb14
from .resnet import (
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    resnet18,
    resnet50,
    resnet152,
)
from .scale_mae import ScaleMAELarge16_Weights, scalemae_large_patch16
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

_model: dict[str, Callable[..., nn.Module]] = {
    'copernicusfm_base': copernicusfm_base,
    'croma_base': croma_base,
    'croma_large': croma_large,
    'dofa_base_patch16_224': dofa_base_patch16_224,
    'dofa_huge_patch14_224': dofa_huge_patch14_224,
    'dofa_large_patch16_224': dofa_large_patch16_224,
    'dofa_small_patch16_224': dofa_small_patch16_224,
    'earthloc': earthloc,
    'panopticon_vitb14': panopticon_vitb14,
    'resnet18': resnet18,
    'resnet50': resnet50,
    'resnet152': resnet152,
    'scalemae_large_patch16': scalemae_large_patch16,
    'swin_v2_t': swin_v2_t,
    'swin_v2_b': swin_v2_b,
    'unet': unet,
    'vit_small_patch16_224': vit_small_patch16_224,
    'vit_base_patch14_dinov2': vit_base_patch14_dinov2,
    'vit_base_patch16_224': vit_base_patch16_224,
    'vit_huge_patch14_224': vit_huge_patch14_224,
    'vit_large_patch16_224': vit_large_patch16_224,
    'vit_small_patch14_dinov2': vit_small_patch14_dinov2,
    'yolo': yolo,
}

_model_weights: dict[str | Callable[..., nn.Module], WeightsEnum] = {
    copernicusfm_base: CopernicusFM_Base_Weights,
    croma_base: CROMABase_Weights,
    croma_large: CROMALarge_Weights,
    dofa_base_patch16_224: DOFABase16_Weights,
    dofa_large_patch16_224: DOFALarge16_Weights,
    earthloc: EarthLoc_Weights,
    panopticon_vitb14: Panopticon_Weights,
    resnet18: ResNet18_Weights,
    resnet50: ResNet50_Weights,
    resnet152: ResNet152_Weights,
    scalemae_large_patch16: ScaleMAELarge16_Weights,
    swin_v2_t: Swin_V2_T_Weights,
    swin_v2_b: Swin_V2_B_Weights,
    unet: Unet_Weights,
    vit_small_patch16_224: ViTSmall16_Weights,
    vit_base_patch14_dinov2: ViTBase14_DINOv2_Weights,
    vit_base_patch16_224: ViTBase16_Weights,
    vit_huge_patch14_224: ViTHuge14_Weights,
    vit_large_patch16_224: ViTLarge16_Weights,
    vit_small_patch14_dinov2: ViTSmall14_DINOv2_Weights,
    yolo: YOLO_Weights,
    'copernicusfm_base': CopernicusFM_Base_Weights,
    'croma_base': CROMABase_Weights,
    'croma_large': CROMALarge_Weights,
    'dofa_base_patch16_224': DOFABase16_Weights,
    'dofa_large_patch16_224': DOFALarge16_Weights,
    'earthloc': EarthLoc_Weights,
    'panopticon_vitb14': Panopticon_Weights,
    'resnet18': ResNet18_Weights,
    'resnet50': ResNet50_Weights,
    'resnet152': ResNet152_Weights,
    'scalemae_large_patch16': ScaleMAELarge16_Weights,
    'swin_v2_t': Swin_V2_T_Weights,
    'swin_v2_b': Swin_V2_B_Weights,
    'unet': Unet_Weights,
    'vit_small_patch16_224': ViTSmall16_Weights,
    'vit_base_patch14_dinov2': ViTBase14_DINOv2_Weights,
    'vit_base_patch16_224': ViTBase16_Weights,
    'vit_huge_patch14_224': ViTHuge14_Weights,
    'vit_large_patch16_224': ViTLarge16_Weights,
    'vit_small_patch14_dinov2': ViTSmall14_DINOv2_Weights,
    'yolo': YOLO_Weights,
}


def get_model(name: str, *args: Any, **kwargs: Any) -> nn.Module:
    """Get an instantiated model from its name.

    .. versionadded:: 0.4

    Args:
        name: Name of the model.
        *args: Additional arguments passed to the model builder method.
        **kwargs: Additional keyword arguments passed to the model builder method.

    Returns:
        An instantiated model.
    """
    model: nn.Module = _model[name](*args, **kwargs)
    return model


def get_model_weights(name: Callable[..., nn.Module] | str) -> WeightsEnum:
    """Get the weights enum class associated with a given model.

    .. versionadded:: 0.4

    Args:
        name: Model builder function or the name under which it is registered.

    Returns:
        The weights enum class associated with the model.
    """
    return _model_weights[name]


def get_weight(name: str) -> WeightsEnum:
    """Get the weights enum value by its full name.

    .. versionadded:: 0.4

    Args:
        name: Name of the weight enum entry.

    Returns:
        The requested weight enum.

    Raises:
        ValueError: If *name* is not a valid WeightsEnum.
    """
    for weight_name, weight_enum in _model_weights.items():
        if isinstance(weight_name, str):
            for sub_weight_enum in weight_enum:
                if name == str(sub_weight_enum):
                    return sub_weight_enum

    raise ValueError(f'{name} is not a valid WeightsEnum')


def list_models() -> list[str]:
    """List the registered models.

    .. versionadded:: 0.4

    Returns:
        A list of registered models.
    """
    return list(_model.keys())
