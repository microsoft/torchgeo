# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import enum
from collections.abc import Callable

import pytest
import torch.nn as nn
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    Aurora_Weights,
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
    Swin_B_Weights,
    Swin_S_Weights,
    Swin_T_Weights,
    Swin_V2_B_Weights,
    Swin_V2_T_Weights,
    Tessera_Weights,
    Unet_Weights,
    ViTBase14_DINOv2_Weights,
    ViTBase16_Weights,
    ViTHuge14_Weights,
    ViTLarge16_Weights,
    ViTSmall14_DINOv2_Weights,
    ViTSmall16_Weights,
    aurora_swin_unet,
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
    swin_b,
    swin_s,
    swin_t,
    swin_v2_b,
    swin_v2_t,
    tessera,
    unet,
    vit_base_patch14_dinov2,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch14_dinov2,
    vit_small_patch16_224,
)

builders = [
    aurora_swin_unet,
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
    swin_t,
    swin_s,
    swin_b,
    swin_v2_t,
    swin_v2_b,
    tessera,
    unet,
    vit_base_patch14_dinov2,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch14_dinov2,
    vit_small_patch16_224,
]
enums = [
    Aurora_Weights,
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
    Swin_T_Weights,
    Swin_S_Weights,
    Swin_B_Weights,
    Swin_V2_T_Weights,
    Swin_V2_B_Weights,
    Tessera_Weights,
    Unet_Weights,
    ViTBase14_DINOv2_Weights,
    ViTBase16_Weights,
    ViTHuge14_Weights,
    ViTLarge16_Weights,
    ViTSmall14_DINOv2_Weights,
    ViTSmall16_Weights,
]


@pytest.mark.parametrize('builder', builders)
def test_get_model(builder: Callable[..., nn.Module]) -> None:
    if builder == aurora_swin_unet:
        pytest.importorskip('aurora')

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
    for weight in enum:  # type: ignore[not-iterable]
        assert weight == get_weight(str(weight))


def test_list_models() -> None:
    models = [builder.__name__ for builder in builders]
    assert set(models) == set(list_models())


def test_invalid_model() -> None:
    with pytest.raises(ValueError, match='bad_model is not a valid WeightsEnum'):
        get_weight('bad_model')
