# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from torchvision.models._api import Weights

from torchgeo.models import (  # noqa: F401
    ResNet18_Weights,
    ResNet50_Weights,
    resnet18,
    resnet50,
)


@pytest.fixture(params=[*ResNet18_Weights, *ResNet50_Weights])
def weights(request: SubRequest) -> Weights:
    return request.param


@pytest.fixture
def mocked_weights(
    tmp_path: Path, monkeypatch: MonkeyPatch, weights: Weights
) -> Weights:
    path = os.path.join(tmp_path, "weight.pth")
    model = timm.create_model(weights.meta["model"], in_chans=weights.meta["in_chans"])
    torch.save(model.state_dict(), path)
    monkeypatch.setattr(weights, "url", "file://" + path)
    return weights


def test_resnet(mocked_weights: Weights) -> None:
    resnet = eval(mocked_weights.meta["model"])
    resnet(weights=mocked_weights)


@pytest.mark.slow
def test_resnet_download(weights: Weights) -> None:
    resnet = eval(weights.meta["model"])
    resnet(weights=weights)
