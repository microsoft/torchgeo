# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
from pytest import MonkeyPatch
from torch import nn

from torchgeo.models.rfdetr import RFDETR


def test_pretrained_weights(monkeypatch: MonkeyPatch) -> None:
    rfdetr = pytest.importorskip('rfdetr')
    calls: list[nn.Module] = []

    monkeypatch.setattr(
        rfdetr.models,
        'build_model_from_config',
        lambda *_args, **_kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        rfdetr.models,
        'build_criterion_from_config',
        lambda *_args, **_kwargs: (nn.Identity(), nn.Identity()),
    )
    monkeypatch.setattr(
        rfdetr.models,
        'load_pretrain_weights',
        lambda model, _config: calls.append(model),
    )

    model = RFDETR(
        'rf-detr-nano',
        num_classes=2,
        in_channels=3,
        freeze_backbone=False,
        pretrain_weights='weights.pth',
    )

    assert calls == [model.model]
