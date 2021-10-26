# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, cast

import pytest
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from pytorch_lightning.core.lightning import LightningModule
from torchvision.models import resnet18

from torchgeo.trainers import BYOLTask, ChesapeakeCVPRDataModule
from torchgeo.trainers.byol import BYOL, SimCLRAugmentation

from .test_utils import mocked_log


@pytest.fixture(scope="module")
def datamodule() -> ChesapeakeCVPRDataModule:
    dm = ChesapeakeCVPRDataModule(
        os.path.join("tests", "data", "chesapeake", "cvpr"),
        ["de-test"],
        ["de-test"],
        ["de-test"],
        patch_size=4,
        patches_per_tile=2,
        batch_size=2,
        num_workers=0,
    )
    dm.prepare_data()
    dm.setup()
    return dm


class TestBYOL:
    def test_custom_augment_fn(self) -> None:
        encoder = resnet18()
        layer = encoder.conv1
        new_layer = nn.Conv2d(  # type: ignore[attr-defined]
            in_channels=4,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias,
        ).requires_grad_()
        encoder.conv1 = new_layer
        augment_fn = SimCLRAugmentation((2, 2))
        BYOL(encoder, augment_fn=augment_fn)


class TestBYOLTask:
    @pytest.fixture(params=["resnet18", "resnet50"])
    def config(self, request: SubRequest) -> Dict[str, Any]:
        task_conf = OmegaConf.load(os.path.join("conf", "task_defaults", "byol.yaml"))
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        task_args["encoder"] = request.param
        return task_args

    @pytest.fixture
    def task(
        self, config: Dict[str, Any], monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> LightningModule:
        task = BYOLTask(**config)
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_configure_optimizers(self, task: BYOLTask) -> None:
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self, datamodule: ChesapeakeCVPRDataModule, task: BYOLTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)

    def test_validation(
        self, datamodule: ChesapeakeCVPRDataModule, task: BYOLTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)

    def test_test(self, datamodule: ChesapeakeCVPRDataModule, task: BYOLTask) -> None:
        batch = next(iter(datamodule.test_dataloader()))
        task.test_step(batch, 0)

    def test_invalid_encoder(self, config: Dict[str, Any]) -> None:
        config["encoder"] = "invalid_encoder"
        error_message = "Encoder type 'invalid_encoder' is not valid."
        with pytest.raises(ValueError, match=error_message):
            BYOLTask(**config)
