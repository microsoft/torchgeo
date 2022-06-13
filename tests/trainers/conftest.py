# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict

import pytest
import torch
import torchvision
from _pytest.fixtures import SubRequest
from torch import Tensor
from torch.nn.modules import Module


@pytest.fixture(scope="package")
def model() -> Module:
    model: Module = torchvision.models.resnet18(pretrained=False)
    return model


@pytest.fixture(scope="package")
def state_dict(model: Module) -> Dict[str, Tensor]:
    return model.state_dict()


@pytest.fixture(params=["classification_model", "encoder_name"])
def checkpoint(
    state_dict: Dict[str, Tensor], request: SubRequest, tmp_path: Path
) -> str:
    if request.param == "classification_model":
        state_dict = OrderedDict({"model." + k: v for k, v in state_dict.items()})
    else:
        state_dict = OrderedDict(
            {"model.encoder.model." + k: v for k, v in state_dict.items()}
        )
    checkpoint = {
        "hyper_parameters": {request.param: "resnet18"},
        "state_dict": state_dict,
    }
    path = os.path.join(str(tmp_path), f"model_{request.param}.ckpt")
    torch.save(checkpoint, path)
    return path
