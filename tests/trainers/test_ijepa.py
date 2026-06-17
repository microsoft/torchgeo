# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any

import pytest
import timm
from pytest import MonkeyPatch
from timm.models import VisionTransformer
from torch.nn import Module

from torchgeo.datasets import SSL4EOS12
from torchgeo.main import main
from torchgeo.trainers import IJEPATask


def create_model(*args: Any, **kwargs: Any) -> Module:
    """Create a tiny ViT for fast testing."""
    kwargs.pop('pretrained', None)
    return VisionTransformer(depth=1, **kwargs)


class TestIJEPATask:
    @pytest.mark.parametrize('name', ['ssl4eo_s12_ijepa_1'])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        monkeypatch.setattr(SSL4EOS12, '__len__', lambda self: 2)
        monkeypatch.setattr(timm, 'create_model', create_model)

        args = [
            '--config',
            config,
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            str(fast_dev_run),
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])

    def test_wrong_model_type(self) -> None:
        with pytest.raises(ValueError, match='is not a ViT architecture'):
            IJEPATask(model='resnet18', weights=None)
