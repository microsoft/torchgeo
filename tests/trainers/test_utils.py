# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.nn.modules import Module

from torchgeo.trainers.utils import extract_encoder, load_state_dict


class FakeExperiment(object):
    def add_figure(self, *args: Any, **kwargs: Any) -> None:
        pass


class FakeLogger(object):
    def __init__(self) -> None:
        self.experiment = FakeExperiment()


class FakeTrainer(object):
    def __init__(self) -> None:
        self.logger = FakeLogger()
        self.global_step = 1


def mocked_log(*args: Any, **kwargs: Any) -> None:
    pass


def test_extract_encoder_unsupported_model(tmp_path: Path) -> None:
    checkpoint = {"hyper_parameters": {"some_unsupported_model": "resnet18"}}
    path = os.path.join(str(tmp_path), "dummy.ckpt")
    torch.save(checkpoint, path)
    err = (
        "Unknown checkpoint task. Only encoder or classification_model"
        " extraction is supported"
    )
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
    model.conv1 = nn.Conv2d(  # type: ignore[attr-defined]
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
    in_features = model.fc.in_features  # type: ignore[union-attr]
    model.fc = nn.Linear(  # type: ignore[attr-defined]
        in_features, out_features=num_classes
    )

    warning = (
        f"num classes {num_classes} != num classes in pretrained model"
        f" {expected_num_classes}. Overriding with new num classes"
    )
    with pytest.warns(UserWarning, match=warning):
        model = load_state_dict(model, state_dict)
