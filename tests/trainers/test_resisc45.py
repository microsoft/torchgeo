# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict

import pytest

from torchgeo.datasets import RESISC45DataModule
from torchgeo.trainers.resisc45 import RESISC45ClassificationTask


class TestRESISC45ClassificationTask:
    @pytest.fixture(scope="class")
    def datamodule() -> RESISC45DataModule:
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
    def task(self, config: Dict[str, Any]) -> RESISC45ClassificationTask:
        task = RESISC45ClassificationTask(**config)
        return task

    def test_training(
        self, datamodule: RESISC45DataModule, task: RESISC45ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)
