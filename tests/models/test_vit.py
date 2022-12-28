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
from torchgeo.models import VITSmall16_Weights
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


# VITSmall16 weights
@pytest.fixture
def vitsmall16_sentinel2_all_moco(tmp_path: Path) -> Tuple[str, int]:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_MOCO"
    model = timm.create_model("vit_small_patch16_224", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"vitsmall16_{weight_key.lower()}.pt")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def vitsmall16_sentinel2_all_dino(tmp_path: Path) -> Tuple[str, int]:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_DINO"
    model = timm.create_model("vit_small_patch16_224", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"vitsmall16_{weight_key.lower()}.pt")
    torch.save({"teacher": model.state_dict()}, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.mark.parametrize(
    "generate_model, weight",
    [
        ("vitsmall16_sentinel2_all_moco", VITSmall16_Weights.SENTINEL2_ALL_MOCO),
        ("vitsmall16_sentinel2_all_dino", VITSmall16_Weights.SENTINEL2_ALL_DINO),
    ],
)
def test_vitsmall16_pretrained_weights(
    monkeypatch: MonkeyPatch, request: SubRequest, generate_model, weight
) -> None:

    ckpt_path, num_input_channels = request.getfixturevalue(generate_model)

    monkeypatch.setattr(weight, "url", ckpt_path)
    monkeypatch.setattr(
        torchgeo.models.vit, "load_state_dict_from_url", load_state_dict_from_url
    )

    task = ClassificationTask(
        model="vit_small_patch16_224",
        loss="ce",
        in_channels=num_input_channels,
        weights=weight.get_state_dict(),
        num_classes=1000,  # imagenet default weights timm
    )
    x = torch.zeros(1, num_input_channels, 224, 224)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)


@pytest.mark.slow
@pytest.mark.parametrize("weight_name", [(w.name) for w in VITSmall16_Weights])
def test_resnet50_weights_download(weight_name: str) -> None:
    weight = VITSmall16_Weights[weight_name]
    state_dict = weight.get_state_dict()
    num_input_channels = weight.meta["num_input_channels"]

    task = ClassificationTask(
        model="vit_small_patch16_224",
        loss="ce",
        in_channels=num_input_channels,
        weights=state_dict,
        num_classes=1000,  # imagenet default weights timm
    )
    x = torch.zeros(1, num_input_channels, 224, 224)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)
