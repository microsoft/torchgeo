# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict

import pytest

from torchgeo.datasets import ETCI2021DataModule
from torchgeo.trainers.etci2021 import ETCI2021SemanticSegmentationTask


class TestETCI2021SemanticSegmentationTask:
    @pytest.fixture(scope="class")
    def datamodule(self) -> ETCI2021DataModule:
        root = os.path.join("tests", "data", "etci2021")
        seed = 0
        batch_size = 2
        num_workers = 0
        dm = ETCI2021DataModule(root, seed, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture()
    def config(self) -> Dict[str, Any]:
        task_args: Dict[str, Any] = {}
        task_args["loss"] = "ce"
        task_args["segmentation_model"] = "unet"
        task_args["encoder_name"] = "resnet50"
        task_args["encoder_weights"] = None
        task_args["learning_rate"] = 3e-4
        task_args["learning_rate_schedule_patience"] = 6
        task_args["in_channels"] = 7
        task_args["num_classes"] = 2
        task_args["weights"] = "random"
        task_args["ignore_zeros"] = True
        return task_args

    @pytest.fixture
    def task(self, config: Dict[str, Any]) -> ETCI2021SemanticSegmentationTask:
        task = ETCI2021SemanticSegmentationTask(**config)
        return task

    def test_training(
        self, datamodule: ETCI2021DataModule, task: ETCI2021SemanticSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)
