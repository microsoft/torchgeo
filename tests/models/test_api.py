# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import enum
from collections.abc import Callable

import pytest
import torch.nn as nn
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    CopernicusFM_Base_Weights,
    CROMABase_Weights,
    CROMALarge_Weights,
    DOFABase16_Weights,
    DOFALarge16_Weights,
    EarthLoc_Weights,
    Panopticon_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    ScaleMAELarge16_Weights,
    Swin_V2_B_Weights,
    Swin_V2_T_Weights,
    Unet_Weights,
    ViTBase14_DINOv2_Weights,
    ViTBase16_Weights,
    ViTHuge14_Weights,
    ViTLarge16_Weights,
    ViTSmall14_DINOv2_Weights,
    ViTSmall16_Weights,
    YOLO_Weights,
    copernicusfm_base,
    croma_base,
    croma_large,
    dofa_base_patch16_224,
    dofa_huge_patch14_224,
    dofa_large_patch16_224,
    dofa_small_patch16_224,
    earthloc,
    get_model,
    get_model_weights,
    get_weight,
    list_models,
    panopticon_vitb14,
    resnet18,
    resnet50,
    resnet152,
    scalemae_large_patch16,
    swin_v2_b,
    swin_v2_t,
    unet,
    vit_base_patch14_dinov2,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch14_dinov2,
    vit_small_patch16_224,
    yolo,
)

builders = [
    copernicusfm_base,
    croma_base,
    croma_large,
    dofa_base_patch16_224,
    dofa_huge_patch14_224,
    dofa_large_patch16_224,
    dofa_small_patch16_224,
    earthloc,
    panopticon_vitb14,
    resnet18,
    resnet50,
    resnet152,
    scalemae_large_patch16,
    swin_v2_t,
    swin_v2_b,
    unet,
    vit_base_patch14_dinov2,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch14_dinov2,
    vit_small_patch16_224,
    yolo,
]
enums = [
    CopernicusFM_Base_Weights,
    CROMABase_Weights,
    CROMALarge_Weights,
    DOFABase16_Weights,
    DOFALarge16_Weights,
    EarthLoc_Weights,
    Panopticon_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    ScaleMAELarge16_Weights,
    Swin_V2_T_Weights,
    Swin_V2_B_Weights,
    Unet_Weights,
    ViTBase14_DINOv2_Weights,
    ViTBase16_Weights,
    ViTHuge14_Weights,
    ViTLarge16_Weights,
    ViTSmall14_DINOv2_Weights,
    ViTSmall16_Weights,
    YOLO_Weights,
]


# check if ultralytics is installed otherwise skip the yolo model tests
try:
    import ultralytics  # noqa: F401

    ULTRALYTICS_INSTALLED = True
except ImportError:
    ULTRALYTICS_INSTALLED = False


@pytest.mark.parametrize('builder', builders)
def test_get_model(builder: Callable[..., nn.Module]) -> None:
    if builder == yolo and not ULTRALYTICS_INSTALLED:
        pytest.skip('Ultralytics is not installed, skipping YOLO model tests')

    model = get_model(builder.__name__)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize('builder', builders)
def test_get_model_weights(builder: Callable[..., nn.Module]) -> None:
    models_without_weights = [dofa_huge_patch14_224, dofa_small_patch16_224]
    if builder in models_without_weights:
        return

    weights = get_model_weights(builder)
    assert isinstance(weights, enum.EnumMeta)
    weights = get_model_weights(builder.__name__)
    assert isinstance(weights, enum.EnumMeta)


@pytest.mark.parametrize('enum', enums)
def test_get_weight(enum: WeightsEnum) -> None:
    for weight in enum:
        assert weight == get_weight(str(weight))


def test_list_models() -> None:
    models = [builder.__name__ for builder in builders]
    assert set(models) == set(list_models())


def test_invalid_model() -> None:
    with pytest.raises(ValueError, match='bad_model is not a valid WeightsEnum'):
        get_weight('bad_model')
