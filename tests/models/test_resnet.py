# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torch import Tensor

import torchgeo.models.resnet
from torchgeo.models import ResNet18_Weights, ResNet50_Weights
from torchgeo.trainers import ClassificationTask


def load_state_dict_from_url(
    root: str, filename: str, url: str, map_location: torch.device
) -> Any:
    """Mockup of ``torchgeo.models.resnet.load_state_dict_from_url."""
    return torch.load(url)


def adjust_moco_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Adjust the moco weight names."""
    new_state_dict = {"module.encoder_q." + key: val for key, val in state_dict.items()}
    return new_state_dict


# RESNET18 Weights
@pytest.fixture
def resnet18_imagenet_rgb(tmp_path: Path) -> Tuple[str, int]:
    num_input_channels = 3
    weight_key = "IMAGENET_RGB"
    model = timm.create_model("resnet18", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet18_{weight_key.lower()}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet18_sentinel2_rgb_moco(tmp_path: Path) -> str:
    num_input_channels = 3
    weight_key = "SENTINEL2_RGB_MOCO"
    model = timm.create_model("resnet18", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet18_{weight_key.lower()}.pt")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet18_sentinel2_all_moco(tmp_path: Path) -> str:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_MOCO"
    model = timm.create_model("resnet18", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet18_{weight_key.lower()}.pt")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.mark.parametrize(
    "generate_model,weight",
    [
        ("resnet18_imagenet_rgb", ResNet18_Weights.IMAGENET_RGB),
        ("resnet18_sentinel2_rgb_moco", ResNet18_Weights.SENTINEL2_RGB_MOCO),
        ("resnet18_sentinel2_all_moco", ResNet18_Weights.SENTINEL2_ALL_MOCO),
    ],
)
def test_resnet18_pretrained_weights(
    monkeypatch: MonkeyPatch, request: SubRequest, generate_model, weight
) -> None:

    ckpt_path, num_input_channels = request.getfixturevalue(generate_model)

    monkeypatch.setattr(weight, "url", ckpt_path)
    monkeypatch.setattr(
        torchgeo.models.resnet, "load_state_dict_from_url", load_state_dict_from_url
    )

    task = ClassificationTask(
        model="resnet18",
        loss="ce",
        in_channels=num_input_channels,
        weights=weight.get_state_dict(),
        num_classes=1000,  # imagenet default weights timm
    )
    x = torch.zeros(1, num_input_channels, 64, 64)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)


# RESNET 50 Weights
@pytest.fixture
def resnet50_googleearth_millionaid_rgb(tmp_path: Path) -> str:
    pytest.importorskip("yacs")
    num_input_channels = 3
    weight_key = "GOOGLEEARTH_MILLIONAID_RGB"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet50_imagenet_rgb(tmp_path: Path) -> str:
    num_input_channels = 3
    weight_key = "IMAGENET_RGB"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet50_sentinel2_rgb_moco(tmp_path: Path) -> str:
    num_input_channels = 3
    weight_key = "SENTINEL2_RGB_MOCO"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pt")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet50_sentinel2_all_moco(tmp_path: Path) -> str:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_MOCO"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pt")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet50_sentinel1_grd_moco(tmp_path: Path) -> str:
    num_input_channels = 2
    weight_key = "SENTINEL1_GRD_MOCO"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pt")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet50_sentinel2_all_dino(tmp_path: Path) -> str:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_DINO"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pt")
    torch.save({"teacher": model.state_dict()}, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.mark.parametrize(
    "generate_model,weight",
    [
        (
            "resnet50_googleearth_millionaid_rgb",
            ResNet50_Weights.GOOGLEEARTH_MILLIONAID_RGB,
        ),
        ("resnet50_imagenet_rgb", ResNet50_Weights.IMAGENET_RGB),
        ("resnet50_sentinel2_rgb_moco", ResNet50_Weights.SENTINEL2_RGB_MOCO),
        ("resnet50_sentinel2_all_moco", ResNet50_Weights.SENTINEL2_ALL_MOCO),
        ("resnet50_sentinel1_grd_moco", ResNet50_Weights.SENTINEL1_GRD_MOCO),
        ("resnet50_sentinel2_all_dino", ResNet50_Weights.SENTINEL2_ALL_DINO),
    ],
)
def test_resnet50_pretrained_weights(
    monkeypatch: MonkeyPatch, request: SubRequest, generate_model, weight
) -> None:

    ckpt_path, num_input_channels = request.getfixturevalue(generate_model)

    monkeypatch.setattr(weight, "url", ckpt_path)
    monkeypatch.setattr(
        torchgeo.models.resnet, "load_state_dict_from_url", load_state_dict_from_url
    )

    task = ClassificationTask(
        model="resnet50",
        loss="ce",
        in_channels=num_input_channels,
        weights=weight.get_state_dict(),
        num_classes=1000,  # imagenet default weights timm
    )
    x = torch.zeros(1, num_input_channels, 64, 64)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)


@pytest.mark.slow
@pytest.mark.parametrize("weight_name", [(w.name) for w in ResNet50_Weights])
def test_resnet50_weights_download(weight_name: str) -> None:
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
