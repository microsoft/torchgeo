# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any

import pytest
import torch
import torch.nn as nn
import torchvision.models.detection
from _pytest.monkeypatch import MonkeyPatch
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from torch.nn.modules import Module

from torchgeo.datamodules import MisconfigurationException, NASAMarineDebrisDataModule
from torchgeo.datasets import NASAMarineDebris
from torchgeo.trainers import ObjectDetectionTask


class PredictObjectDetectionDataModule(NASAMarineDebrisDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = NASAMarineDebris(**self.kwargs)


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
                output.append(
                    {
                        "boxes": torch.rand(10, 4),
                        "labels": torch.randint(0, 2, (10,)),
                        "scores": torch.rand(10),
                    }
                )
            return output


def plot(*args: Any, **kwargs: Any) -> None:
    raise ValueError


class TestObjectDetectionTask:
    @pytest.mark.parametrize("name", ["nasa_marine_debris"])
    @pytest.mark.parametrize("model_name", ["faster-rcnn", "fcos", "retinanet"])
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, model_name: str, fast_dev_run: bool
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", f"{name}.yaml"))

        # Instantiate datamodule
        datamodule = instantiate(conf.datamodule)

        # Instantiate model
        monkeypatch.setattr(
            torchvision.models.detection, "FasterRCNN", ObjectDetectionTestModel
        )
        monkeypatch.setattr(
            torchvision.models.detection, "FCOS", ObjectDetectionTestModel
        )
        monkeypatch.setattr(
            torchvision.models.detection, "RetinaNet", ObjectDetectionTestModel
        )
        conf.module.model = model_name
        model = instantiate(conf.module)

        # Instantiate trainer
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model=model, datamodule=datamodule)
        try:
            trainer.test(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass
        try:
            trainer.predict(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass

    @pytest.fixture
    def model_kwargs(self) -> dict[Any, Any]:
        return {"model": "faster-rcnn", "backbone": "resnet18", "num_classes": 2}

    def test_invalid_model(self, model_kwargs: dict[Any, Any]) -> None:
        model_kwargs["model"] = "invalid_model"
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetectionTask(**model_kwargs)

    def test_invalid_backbone(self, model_kwargs: dict[Any, Any]) -> None:
        model_kwargs["backbone"] = "invalid_backbone"
        match = "Backbone type 'invalid_backbone' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetectionTask(**model_kwargs)

    def test_non_pretrained_backbone(self, model_kwargs: dict[Any, Any]) -> None:
        model_kwargs["pretrained"] = False
        ObjectDetectionTask(**model_kwargs)

    def test_no_rgb(
        self, monkeypatch: MonkeyPatch, model_kwargs: dict[Any, Any], fast_dev_run: bool
    ) -> None:
        monkeypatch.setattr(NASAMarineDebrisDataModule, "plot", plot)
        datamodule = NASAMarineDebrisDataModule(
            root="tests/data/nasa_marine_debris", batch_size=1, num_workers=0
        )
        model = ObjectDetectionTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, model_kwargs: dict[Any, Any], fast_dev_run: bool) -> None:
        datamodule = PredictObjectDetectionDataModule(
            root="tests/data/nasa_marine_debris", batch_size=1, num_workers=0
        )
        model = ObjectDetectionTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    @pytest.mark.parametrize("model_name", ["faster-rcnn", "fcos", "retinanet"])
    def test_freeze_backbone(
        self, model_name: str, model_kwargs: dict[Any, Any]
    ) -> None:
        model_kwargs["freeze_backbone"] = True
        model_kwargs["model"] = model_name
        model = ObjectDetectionTask(**model_kwargs)
        assert not all([param.requires_grad for param in model.model.parameters()])
