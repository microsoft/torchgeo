# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, cast

import pytest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.datasets import LandCoverAIDataModule
from torchgeo.trainers.landcoverai import LandCoverAISegmentationTask

from .test_utils import FakeTrainer, mocked_log


class TestLandCoverAISegmentationTask:
    @pytest.fixture(scope="class")
    def datamodule(self) -> LandCoverAIDataModule:
        root = os.path.join("tests", "data", "landcoverai")
        batch_size = 2
        num_workers = 0
        dm = LandCoverAIDataModule(root, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "landcoverai.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        task_args["verbose"] = True
        return task_args

    @pytest.fixture
    def task(
        self, config: Dict[str, Any], monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> LandCoverAISegmentationTask:
        task = LandCoverAISegmentationTask(**config)
        trainer = FakeTrainer()
        monkeypatch.setattr(task, "trainer", trainer)  # type: ignore[attr-defined]
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_training(
        self, datamodule: LandCoverAIDataModule, task: LandCoverAISegmentationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self, datamodule: LandCoverAIDataModule, task: LandCoverAISegmentationTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self, datamodule: LandCoverAIDataModule, task: LandCoverAISegmentationTask
    ) -> None:
        batch = next(iter(datamodule.test_dataloader()))
        task.test_step(batch, 0)
        task.test_epoch_end(0)
