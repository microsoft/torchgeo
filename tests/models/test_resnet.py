# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch

import torchgeo.models.resnet
from torchgeo.models import ResNet50_Weights
from torchgeo.trainers import ClassificationTask


def load_state_dict_from_url(
    root: str, filename: str, url: str, map_location: torch.device
) -> Any:
    """Mockup of ``torchgeo.models.resnet.load_state_dict_from_url."""
    return torch.load(url)


@pytest.fixture
def resnet50_googleearth_millionaid_rgb(tmp_path: Path) -> str:
    num_input_channels = 3
    weight_key = "GOOGLEEARTH_MILLIONAID_RGB"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path, num_input_channels, weight_key


@pytest.fixture
def resnet50_imagenet_rgb(tmp_path: Path) -> str:
    num_input_channels = 3
    weight_key = "IMAGENET_RGB"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path, num_input_channels, weight_key


@pytest.mark.parametrize(
    "weights", [("resnet50_googleearth_millionaid_rgb"), ("resnet50_imagenet_rgb")]
)
def test_resnet50_pretrained_weights(
    monkeypatch: MonkeyPatch, request: SubRequest, weights
) -> None:

    ckpt_path, num_input_channels, weight_key = request.getfixturevalue(weights)

    monkeypatch.setattr(
        torchgeo.models.ResNet50_Weights, f"{weight_key}.url", ckpt_path
    )
    monkeypatch.setattr(
        torchgeo.models.resnet, "load_state_dict_from_url", load_state_dict_from_url
    )

    task = ClassificationTask(
        model="resnet50",
        loss="ce",
        in_channels=num_input_channels,
        weights=ResNet50_Weights[weight_key].get_state_dict(),
        num_classes=1000,  # imagenet default weights timm
    )
    x = torch.zeros(1, num_input_channels, 64, 64)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)


@pytest.mark.parametrize("weight_name", [(w.name) for w in ResNet50_Weights])
def test_resnet50_weights_download(weight_name=str) -> None:
    weight = ResNet50_Weights[weight_name]
    state_dict = weight.get_state_dict()
    num_input_channels = weight.meta["num_input_channels"]

    task = ClassificationTask(
        model="resnet50",
        loss="ce",
        in_channels=num_input_channels,
        weights=state_dict,
        num_classes=1000,  # imagenet default weights timm
    )
    x = torch.zeros(1, num_input_channels, 64, 64)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)
