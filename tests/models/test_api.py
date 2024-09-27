# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import enum
from collections.abc import Callable

import pytest
import torch.nn as nn
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    DOFABase16_Weights,
    DOFALarge16_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    ScaleMAELarge16_Weights,
    Swin_V2_B_Weights,
    Swin_V2_T_Weights,
    ViTSmall16_Weights,
    dofa_base_patch16_224,
    dofa_large_patch16_224,
    get_model,
    get_model_weights,
    get_weight,
    list_models,
    resnet18,
    resnet50,
    resnet152,
    scalemae_large_patch16,
    swin_v2_b,
    swin_v2_t,
    vit_small_patch16_224,
)

builders = [
    dofa_base_patch16_224,
    dofa_large_patch16_224,
    resnet18,
    resnet50,
    resnet152,
    scalemae_large_patch16,
    swin_v2_t,
    swin_v2_b,
    vit_small_patch16_224,
]
enums = [
    DOFABase16_Weights,
    DOFALarge16_Weights,
    ResNet18_Weights,
    ResNet50_Weights,
    ResNet152_Weights,
    ScaleMAELarge16_Weights,
    Swin_V2_T_Weights,
    Swin_V2_B_Weights,
    ViTSmall16_Weights,
]


@pytest.mark.parametrize('builder', builders)
def test_get_model(builder: Callable[..., nn.Module]) -> None:
    model = get_model(builder.__name__)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize('builder', builders)
def test_get_model_weights(builder: Callable[..., nn.Module]) -> None:
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
