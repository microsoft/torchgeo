# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import enum
from typing import Callable

import pytest
import torch.nn as nn
from torchvision.models._api import WeightsEnum

from torchgeo.models import (
    ResNet18_Weights,
    ResNet50_Weights,
    Swin_V2_B_Weights,
    ViTSmall16_Weights,
    get_model,
    get_model_weights,
    get_weight,
    list_models,
    resnet18,
    resnet50,
    swin_v2_b,
    vit_small_patch16_224,
)

builders = [resnet18, resnet50, vit_small_patch16_224, swin_v2_b]
enums = [ResNet18_Weights, ResNet50_Weights, ViTSmall16_Weights, Swin_V2_B_Weights]


@pytest.mark.parametrize("builder", builders)
def test_get_model(builder: Callable[..., nn.Module]) -> None:
    model = get_model(builder.__name__)
    assert isinstance(model, nn.Module)


@pytest.mark.parametrize("builder", builders)
def test_get_model_weights(builder: Callable[..., nn.Module]) -> None:
    weights = get_model_weights(builder)
    assert isinstance(weights, enum.EnumMeta)
    weights = get_model_weights(builder.__name__)
    assert isinstance(weights, enum.EnumMeta)


@pytest.mark.parametrize("enum", enums)
def test_get_weight(enum: WeightsEnum) -> None:
    for weight in enum:
        assert weight == get_weight(str(weight))


def test_list_models() -> None:
    models = [builder.__name__ for builder in builders]
    assert set(models) == set(list_models())
