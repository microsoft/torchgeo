# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import enum
from collections.abc import Callable
from pathlib import Path

import pytest
import timm
import torch
import torch.nn as nn
from pytest import MonkeyPatch

from torchgeo.models import (
    Aurora_Weights,
    CopernicusFM_Base_Weights,
    CROMABase_Weights,
    CROMALarge_Weights,
    DOFABase16_Weights,
    DOFALarge16_Weights,
    EarthLoc_Weights,
    OlmoEarthV1_Weights,
    Panopticon_Weights,
    Presto_Weights,
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
    TileNet_Weights,
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
    olmoearth_v1,
    panopticon_vitb14,
    presto,
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
    tilenet,
    unet,
    vit_base_patch14_dinov2,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch14_dinov2,
    vit_small_patch16_224,
)
from torchgeo.models._weights import WeightsEnum

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
    olmoearth_v1,
    panopticon_vitb14,
    presto,
    resnet18,
    resnet50,
    resnet152,
    scalemae_large_patch16,
    swin_t,
    swin_s,
    swin_b,
    swin_v2_t,
    swin_v2_b,
    tilenet,
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
    OlmoEarthV1_Weights,
    Panopticon_Weights,
    Presto_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    ScaleMAELarge16_Weights,
    Swin_T_Weights,
    Swin_S_Weights,
    Swin_B_Weights,
    Swin_V2_T_Weights,
    Swin_V2_B_Weights,
    TileNet_Weights,
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
    if builder == olmoearth_v1:
        pytest.importorskip('olmoearth_pretrain_minimal')

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
    for weight in enum:  # ty: ignore[not-iterable]
        assert weight == get_weight(str(weight))


def test_list_models() -> None:
    models = [builder.__name__ for builder in builders]
    assert set(models) == set(list_models())


def test_timm_registry(
    tmp_path: Path, load_state_dict_from_url: None, monkeypatch: MonkeyPatch
) -> None:
    model_name = 'torchgeo_resnet18.sentinel2_all_moco'
    assert model_name in timm.list_models(pretrained=True, include_tags=True)

    cfg = timm.get_pretrained_cfg(model_name)
    assert cfg is not None
    assert cfg.input_size == (13, 224, 224)
    assert cfg.meta['model'] == 'resnet18'

    model = timm.create_model('torchgeo_resnet18', in_chans=13)
    assert isinstance(model, nn.Module)

    weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
    path = tmp_path / 'weights.pth'
    torch.save(timm.create_model('resnet18', in_chans=13).state_dict(), path)
    monkeypatch.setattr(weights.value, 'url', str(path))
    model = timm.create_model(model_name, pretrained=True)
    assert model.conv1.in_channels == 13


def test_invalid_model() -> None:
    with pytest.raises(ValueError, match='bad_model is not a valid WeightsEnum'):
        get_weight('bad_model')
