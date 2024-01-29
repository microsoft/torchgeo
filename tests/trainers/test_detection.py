# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any

import pytest
import torch
import torch.nn as nn
import torchvision.models.detection
from lightning.pytorch import Trainer
from pytest import MonkeyPatch
from torch.nn.modules import Module

from torchgeo.datamodules import MisconfigurationException, NASAMarineDebrisDataModule
from torchgeo.datasets import NASAMarineDebris, RGBBandsMissingError
from torchgeo.main import main
from torchgeo.trainers import ObjectDetectionTask

# MAP metric requires pycocotools to be installed
pytest.importorskip("pycocotools")


class PredictObjectDetectionDataModule(NASAMarineDebrisDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = NASAMarineDebris(**self.kwargs)


def plot_missing_bands(*args: Any, **kwargs: Any) -> None:
    raise RGBBandsMissingError()


class ObjectDetectionTestModel(Module):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.fc = nn.Linear(1, 1)

    def forward(self, images: Any, targets: Any = None) -> Any:
        batch_size = len(images)
        if self.training:
            assert batch_size == len(targets)
            # use the Linear layer to generate a tensor that has a gradient
            return {
                "loss_classifier": self.fc(torch.rand(1)),
                "loss_box_reg": self.fc(torch.rand(1)),
                "loss_objectness": self.fc(torch.rand(1)),
                "loss_rpn_box_reg": self.fc(torch.rand(1)),
            }
        else:  # eval mode
            output = []
            for i in range(batch_size):
                boxes = torch.rand(10, 4)
                # xmax, ymax must be larger than xmin, ymin
                boxes[:, 2:] += 1
                output.append(
                    {
                        "boxes": boxes,
                        "labels": torch.randint(0, 2, (10,)),
                        "scores": torch.rand(10),
                    }
                )
            return output


def plot(*args: Any, **kwargs: Any) -> None:
    return None


class TestObjectDetectionTask:
    @pytest.mark.parametrize("name", ["nasa_marine_debris", "vhr10"])
    @pytest.mark.parametrize("model_name", ["faster-rcnn", "fcos", "retinanet"])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, model_name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join("tests", "conf", name + ".yaml")

        monkeypatch.setattr(
            torchvision.models.detection, "FasterRCNN", ObjectDetectionTestModel
        )
        monkeypatch.setattr(
            torchvision.models.detection, "FCOS", ObjectDetectionTestModel
        )
        monkeypatch.setattr(
            torchvision.models.detection, "RetinaNet", ObjectDetectionTestModel
        )

        args = [
            "--config",
            config,
            "--trainer.accelerator",
            "cpu",
            "--trainer.fast_dev_run",
            str(fast_dev_run),
            "--trainer.max_epochs",
            "1",
            "--trainer.log_every_n_steps",
            "1",
        ]

        main(["fit"] + args)
        try:
            main(["test"] + args)
        except MisconfigurationException:
            pass
        try:
            main(["predict"] + args)
        except MisconfigurationException:
            pass

    def test_invalid_model(self) -> None:
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetectionTask(model="invalid_model")

    def test_invalid_backbone(self) -> None:
        match = "Backbone type 'invalid_backbone' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetectionTask(backbone="invalid_backbone")

    def test_pretrained_backbone(self) -> None:
        ObjectDetectionTask(backbone="resnet18", weights=True)

    def test_no_plot_method(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(NASAMarineDebrisDataModule, "plot", plot)
        datamodule = NASAMarineDebrisDataModule(
            root="tests/data/nasa_marine_debris", batch_size=1, num_workers=0
        )
        model = ObjectDetectionTask(backbone="resnet18", num_classes=2)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(NASAMarineDebrisDataModule, "plot", plot_missing_bands)
        datamodule = NASAMarineDebrisDataModule(
            root="tests/data/nasa_marine_debris", batch_size=1, num_workers=0
        )
        model = ObjectDetectionTask(backbone="resnet18", num_classes=2)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictObjectDetectionDataModule(
            root="tests/data/nasa_marine_debris", batch_size=1, num_workers=0
        )
        model = ObjectDetectionTask(backbone="resnet18", num_classes=2)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    @pytest.mark.parametrize("model_name", ["faster-rcnn", "fcos", "retinanet"])
    def test_freeze_backbone(self, model_name: str) -> None:
        model = ObjectDetectionTask(
            model=model_name, backbone="resnet18", freeze_backbone=True
        )
        assert not all([param.requires_grad for param in model.model.parameters()])
