# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator

import pytest
from _pytest.monkeypatch import MonkeyPatch

from torchgeo.datamodules import RESISC45DataModule
from torchgeo.trainers.resisc45 import RESISC45ClassificationTask

from .test_utils import FakeTrainer, mocked_log


class TestRESISC45ClassificationTask:
    @pytest.fixture(scope="class")
    def datamodule(self) -> RESISC45DataModule:
        root = os.path.join("tests", "data", "resisc45")
        batch_size = 2
        num_workers = 0
        dm = RESISC45DataModule(root, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture()
    def config(self) -> Dict[str, Any]:
        task_args: Dict[str, Any] = {}
        task_args["classification_model"] = "resnet18"
        task_args["learning_rate"] = 3e-4
        task_args["learning_rate_schedule_patience"] = 6
        task_args["in_channels"] = 3
        task_args["loss"] = "ce"
        task_args["num_classes"] = 45
        task_args["weights"] = "random"
        return task_args

    @pytest.fixture
    def task(
        self, config: Dict[str, Any], monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> RESISC45ClassificationTask:
        task = RESISC45ClassificationTask(**config)
        trainer = FakeTrainer()
        monkeypatch.setattr(task, "trainer", trainer)  # type: ignore[attr-defined]
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_training(
        self, datamodule: RESISC45DataModule, task: RESISC45ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)
