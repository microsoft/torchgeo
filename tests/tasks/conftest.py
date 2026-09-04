# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from collections import OrderedDict
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
import yaml
from _pytest.fixtures import SubRequest
from torch import Tensor
from torch.nn.modules import Module


@pytest.fixture(params=[True, pytest.param(False, marks=pytest.mark.slow)])
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


@pytest.fixture(scope='session')
def test_config(
    test_data: Callable[[str], str], tmp_path_factory: pytest.TempPathFactory
) -> Callable[[str], str]:
    """Resolve training configuration paths against the generated fake data."""
    root = tmp_path_factory.mktemp('test-configs')

    def resolve(value: Any) -> Any:
        if isinstance(value, str) and value.startswith('tests/data/'):
            return test_data(value.removeprefix('tests/data/'))
        if isinstance(value, dict):
            return {key: resolve(item) for key, item in value.items()}
        if isinstance(value, list):
            return [resolve(item) for item in value]
        return value

    @cache
    def generate(name: str) -> str:
        source = Path(__file__).parents[1] / 'conf' / name
        config = resolve(yaml.safe_load(source.read_text()))
        destination = root / name
        destination.write_text(yaml.safe_dump(config))
        return str(destination)

    return generate
