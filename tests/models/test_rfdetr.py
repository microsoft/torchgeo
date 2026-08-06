# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import sys
from types import ModuleType

from pytest import MonkeyPatch
from torch import nn

from torchgeo.models.rfdetr import RFDETR


def patch_rfdetr(monkeypatch: MonkeyPatch) -> dict[str, list[object]]:
    calls: dict[str, list[object]] = {'load_pretrain_weights': [], 'apply_lora': []}

    class ModelConfig:
        def __init__(self, num_classes: int, **kwargs: object) -> None:
            self.num_classes = num_classes
            self.pretrain_weights = kwargs.get('pretrain_weights')
            self.backbone_lora = bool(kwargs.get('backbone_lora'))

    class TrainConfig:
        def __init__(self, dataset_dir: str, output_dir: str) -> None:
            self.dataset_dir = dataset_dir
            self.output_dir = output_dir

    def build_model_from_config(
        model_config: object, train_config: object
    ) -> nn.Module:
        return nn.Identity()

    def load_pretrain_weights(model: nn.Module, model_config: object) -> None:
        calls['load_pretrain_weights'].append(model)

    def apply_lora(model: nn.Module) -> None:
        calls['apply_lora'].append(model)

    def build_criterion_from_config(
        model_config: object, train_config: object
    ) -> tuple[object, object]:
        return object(), object()

    rfdetr = ModuleType('rfdetr')
    config = ModuleType('rfdetr.config')
    models = ModuleType('rfdetr.models')
    config.RFDETRLargeConfig = ModelConfig
    config.RFDETRMediumConfig = ModelConfig
    config.RFDETRNanoConfig = ModelConfig
    config.RFDETRSmallConfig = ModelConfig
    config.TrainConfig = TrainConfig
    models.apply_lora = apply_lora
    models.build_criterion_from_config = build_criterion_from_config
    models.build_model_from_config = build_model_from_config
    models.load_pretrain_weights = load_pretrain_weights
    rfdetr.config = config
    rfdetr.models = models
    monkeypatch.setitem(sys.modules, 'rfdetr', rfdetr)
    monkeypatch.setitem(sys.modules, 'rfdetr.config', config)
    monkeypatch.setitem(sys.modules, 'rfdetr.models', models)
    return calls


def test_pretrained_lora(monkeypatch: MonkeyPatch) -> None:
    calls = patch_rfdetr(monkeypatch)
    model = RFDETR(
        'rf-detr-nano',
        num_classes=2,
        in_channels=3,
        freeze_backbone=False,
        pretrain_weights='weights.pth',
        backbone_lora=True,
    )
    assert calls['load_pretrain_weights'] == [model.model]
    assert calls['apply_lora'] == [model.model]
