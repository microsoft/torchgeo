# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any

import pytest
import torch
import torch.nn as nn
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch.nn.modules import Module

from torchgeo.datamodules import MisconfigurationException, VHR10DataModule
from torchgeo.datasets import VHR10, RGBBandsMissingError
from torchgeo.main import main
from torchgeo.trainers import InstanceSegmentationTask

# MAP metric requires pycocotools to be installed
pytest.importorskip('pycocotools')


class PredictInstanceSegmentationDataModule(VHR10DataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = VHR10(**self.kwargs)


# TODO: This is not even used yet
class InstanceSegmentationTestModel(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, images: Any, targets: Any = None) -> Any:
        batch_size = len(images)
        if self.training:
            assert batch_size == len(targets)
            # use the Linear layer to generate a tensor that has a gradient
            return {
                'loss_classifier': self.fc(torch.rand(1)),
                'loss_box_reg': self.fc(torch.rand(1)),
                'loss_objectness': self.fc(torch.rand(1)),
                'loss_rpn_box_reg': self.fc(torch.rand(1)),
            }
        else:  # eval mode
            output = []
            for i in range(batch_size):
                boxes = torch.rand(10, 4)
                # xmax, ymax must be larger than xmin, ymin
                boxes[:, 2:] += 1
                output.append(
                    {
                        'boxes': boxes,
                        'masks': torch.randint(2, images.shape[1:]),
                        'labels': torch.randint(2, (10,)),
                        'scores': torch.rand(10),
                    }
                )
            return output


def plot(*args: Any, **kwargs: Any) -> None:
    return None


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


class TestInstanceSegmentationTask:
    @pytest.mark.parametrize('name', ['vhr10_ins_seg'])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

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
        try:
            main(['test', *args])
        except MisconfigurationException:
            pass
        try:
            main(['predict', *args])
        except MisconfigurationException:
            pass

    def test_invalid_model(self) -> None:
        match = 'Invalid model type'
        with pytest.raises(ValueError, match=match):
            InstanceSegmentationTask(model='invalid_model')

    def test_invalid_backbone(self) -> None:
        match = 'Invalid backbone type'
        with pytest.raises(ValueError, match=match):
            InstanceSegmentationTask(backbone='invalid_backbone')

    def test_pretrained_backbone(self) -> None:
        InstanceSegmentationTask(backbone='resnet50', weights=True)

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(VHR10DataModule, 'plot', plot)
        datamodule = VHR10DataModule(
            root='tests/data/vhr10', batch_size=1, num_workers=0
        )
        model = InstanceSegmentationTask(in_channels=3, num_classes=11)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(VHR10DataModule, 'plot', plot_missing_bands)
        datamodule = VHR10DataModule(
            root='tests/data/vhr10', batch_size=1, num_workers=0
        )
        model = InstanceSegmentationTask(in_channels=3, num_classes=11)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictInstanceSegmentationDataModule(
            root='tests/data/vhr10', batch_size=1, num_workers=0
        )
        model = InstanceSegmentationTask(num_classes=11)
        trainer = Trainer(
            accelerator='cpu',
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    def test_freeze_backbone(self) -> None:
        task = InstanceSegmentationTask(freeze_backbone=True)
        for param in task.model.backbone.parameters():
            assert param.requires_grad is False

        for head in ['rpn', 'roi_heads']:
            for param in getattr(task.model, head).parameters():
                assert param.requires_grad is True
