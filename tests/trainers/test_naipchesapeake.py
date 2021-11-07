# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, cast

import pytest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.datasets import NAIPChesapeakeDataModule
from torchgeo.trainers.naipchesapeake import NAIPChesapeakeSegmentationTask

from .test_utils import FakeTrainer, mocked_log


class TestNAIPChesapeakeSegmentationTask:
    @pytest.fixture(scope="class")
    def datamodule(self) -> NAIPChesapeakeDataModule:
        dm = NAIPChesapeakeDataModule(
            os.path.join("tests", "data", "naip"),
            os.path.join("tests", "data", "chesapeake", "BAYWIDE"),
            batch_size=2,
            num_workers=0,
        )
        dm.patch_size = 32
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "naipchesapeake.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def task(
        self, config: Dict[str, Any], monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> NAIPChesapeakeSegmentationTask:
        task = NAIPChesapeakeSegmentationTask(**config)
        trainer = FakeTrainer()
        monkeypatch.setattr(task, "trainer", trainer)  # type: ignore[attr-defined]
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_validation(
        self, datamodule: NAIPChesapeakeDataModule, task: NAIPChesapeakeSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)
