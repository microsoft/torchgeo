# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, cast

import pytest
from _pytest.fixtures import SubRequest
from omegaconf import OmegaConf
from pytorch_lightning.core.lightning import LightningModule

from torchgeo.trainers import BYOLTask, ChesapeakeCVPRDataModule


class TestBYOLTask:
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load(os.path.join("conf", "task_defaults", "byol.yaml"))
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def datamodule(self) -> ChesapeakeCVPRDataModule:
        dm = ChesapeakeCVPRDataModule(
            os.path.join("tests", "data", "chesapeake", "cvpr"),
            ["de-test"],
            ["de-test"],
            ["de-test"],
            patch_size=128,
            patches_per_tile=2,
            batch_size=2,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture(params=["resnet18", "resnet50"])
    def task(self, config: Dict[str, Any], request: SubRequest) -> LightningModule:
        config["encoder"] = request.param
        return BYOLTask(**config)

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
