# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import cast

import pytest
import timm
import torch
import torch.nn as nn
from pytest import MonkeyPatch
from torch.nn.modules import Module

from torchgeo.trainers.utils import (
    _get_input_layer_name_and_module,
    extract_backbone,
    load_state_dict,
    reinit_initial_conv_layer,
)


def test_extract_backbone(checkpoint: str) -> None:
    extract_backbone(checkpoint)


def test_extract_backbone_unsupported_model(tmp_path: Path) -> None:
    checkpoint = {'hyper_parameters': {'some_unsupported_model': 'resnet18'}}
    path = os.path.join(str(tmp_path), 'dummy.ckpt')
    torch.save(checkpoint, path)
    err = 'Unknown checkpoint task. Only backbone or model extraction is supported'
    with pytest.raises(ValueError, match=err):
        extract_backbone(path)


def test_get_input_layer_name_and_module() -> None:
    key, module = _get_input_layer_name_and_module(timm.create_model('resnet18'))  # type: ignore[attr-defined]
    assert key == 'conv1'
    assert isinstance(module, nn.Conv2d)
    assert module.in_channels == 3


def test_load_state_dict(checkpoint: str, model: Module) -> None:
    _, state_dict = extract_backbone(checkpoint)
    load_state_dict(model, state_dict)


def test_load_state_dict_unequal_input_channels(
    monkeypatch: MonkeyPatch, checkpoint: str, model: Module
) -> None:
    _, state_dict = extract_backbone(checkpoint)
    expected_in_channels = state_dict['conv1.weight'].shape[1]

    in_channels = 7
    conv1 = nn.Conv2d(
        in_channels, out_channels=64, kernel_size=7, stride=1, padding=2, bias=False
    )
    monkeypatch.setattr(model, 'conv1', conv1)

    warning = (
        f'input channels {in_channels} != input channels in pretrained'
        f' model {expected_in_channels}. Overriding with new input channels'
    )
    with pytest.warns(UserWarning, match=warning):
        load_state_dict(model, state_dict)


def test_load_state_dict_unequal_classes(
    monkeypatch: MonkeyPatch, checkpoint: str, model: Module
) -> None:
    _, state_dict = extract_backbone(checkpoint)
    expected_num_classes = state_dict['fc.weight'].shape[0]

    num_classes = 10
    in_features = cast(int, cast(nn.Module, model.fc).in_features)
    fc = nn.Linear(in_features, out_features=num_classes)
    monkeypatch.setattr(model, 'fc', fc)

    warning = (
        f'num classes {num_classes} != num classes in pretrained model'
        f' {expected_num_classes}. Overriding with new num classes'
    )
    with pytest.warns(UserWarning, match=warning):
        load_state_dict(model, state_dict)


def test_reinit_initial_conv_layer() -> None:
    conv_layer = nn.Conv2d(3, 5, kernel_size=3, stride=2, padding=1, bias=True)
    initial_weights = conv_layer.weight.data.clone()

    new_conv_layer = reinit_initial_conv_layer(conv_layer, 4, keep_rgb_weights=True)

    out_channels, in_channels, k1, k2 = new_conv_layer.weight.shape
    assert torch.allclose(initial_weights, new_conv_layer.weight.data[:, :3, :, :])
    assert out_channels == 5
    assert in_channels == 4
    assert k1 == 3 and k2 == 3
    assert new_conv_layer.stride[0] == 2
