# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.nn as nn
from torch.nn.modules import Module

from torchgeo.trainers.utils import (
    extract_encoder,
    load_state_dict,
    reinit_initial_conv_layer,
)


class ClassificationTestModel(Module):
    def __init__(
        self, in_chans: int = 3, num_classes: int = 1000, **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class RegressionTestModel(ClassificationTestModel):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(in_chans=3, num_classes=1)


class SegmentationTestModel(Module):
    def __init__(
        self, in_channels: int = 3, classes: int = 1000, **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


def test_extract_encoder_unsupported_model(tmp_path: Path) -> None:
    checkpoint = {"hyper_parameters": {"some_unsupported_model": "resnet18"}}
    path = os.path.join(str(tmp_path), "dummy.ckpt")
    torch.save(checkpoint, path)
    err = "Unknown checkpoint task. Only encoder or model extraction is supported"
    with pytest.raises(ValueError, match=err):
        extract_encoder(path)


def test_extract_encoder(checkpoint: str) -> None:
    extract_encoder(checkpoint)


def test_load_state_dict(checkpoint: str, model: Module) -> None:
    _, state_dict = extract_encoder(checkpoint)
    model = load_state_dict(model, state_dict)


def test_load_state_dict_unequal_input_channels(checkpoint: str, model: Module) -> None:
    _, state_dict = extract_encoder(checkpoint)
    expected_in_channels = state_dict["conv1.weight"].shape[1]

    in_channels = 7
    model.conv1 = nn.Conv2d(
        in_channels, out_channels=64, kernel_size=7, stride=1, padding=2, bias=False
    )

    warning = (
        f"input channels {in_channels} != input channels in pretrained"
        f" model {expected_in_channels}. Overriding with new input channels"
    )
    with pytest.warns(UserWarning, match=warning):
        model = load_state_dict(model, state_dict)


def test_load_state_dict_unequal_classes(checkpoint: str, model: Module) -> None:
    _, state_dict = extract_encoder(checkpoint)
    expected_num_classes = state_dict["fc.weight"].shape[0]

    num_classes = 10
    in_features = cast(int, cast(nn.Module, model.fc).in_features)
    model.fc = nn.Linear(in_features, out_features=num_classes)

    warning = (
        f"num classes {num_classes} != num classes in pretrained model"
        f" {expected_num_classes}. Overriding with new num classes"
    )
    with pytest.warns(UserWarning, match=warning):
        model = load_state_dict(model, state_dict)


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
