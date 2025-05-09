# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from pathlib import Path

import pytest
import timm
import torch
from _pytest.fixtures import SubRequest
from torch import Tensor
from torch.nn.modules import Module


@pytest.fixture(
    scope='package', params=[True, pytest.param(False, marks=pytest.mark.slow)]
)
def fast_dev_run(request: SubRequest) -> bool:
    flag: bool = request.param
    return flag


@pytest.fixture(scope='package')
def model(request: SubRequest) -> Module:
    in_channels = getattr(request, 'param', 3)
    model: Module = timm.create_model('resnet18', in_chans=in_channels)
    return model


@pytest.fixture(scope='package')
def state_dict(model: Module) -> dict[str, Tensor]:
    return model.state_dict()


@pytest.fixture(params=['model', 'backbone'])
def checkpoint(
    state_dict: dict[str, Tensor], request: SubRequest, tmp_path: Path
) -> str:
    if request.param == 'model':
        state_dict = OrderedDict({'model.' + k: v for k, v in state_dict.items()})
    else:
        state_dict = OrderedDict(
            {'model.backbone.model.' + k: v for k, v in state_dict.items()}
        )
    checkpoint = {
        'hyper_parameters': {request.param: 'resnet18'},
        'state_dict': state_dict,
    }
    path = os.path.join(str(tmp_path), f'model_{request.param}.ckpt')
    torch.save(checkpoint, path)
    return path
