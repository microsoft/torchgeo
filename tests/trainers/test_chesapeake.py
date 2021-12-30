# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, cast

import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.datamodules import ChesapeakeCVPRDataModule
from torchgeo.trainers.chesapeake import ChesapeakeCVPRSegmentationTask

from .test_utils import FakeTrainer, mocked_log


class TestChesapeakeCVPRSegmentationTask:
    @pytest.fixture(scope="class", params=[5, 7])
    def class_set(self, request: SubRequest) -> int:
        return cast(int, request.param)

    @pytest.fixture(scope="class")
    def datamodule(self, class_set: int) -> ChesapeakeCVPRDataModule:
        dm = ChesapeakeCVPRDataModule(
            os.path.join("tests", "data", "chesapeake", "cvpr"),
            ["de-test"],
            ["de-test"],
            ["de-test"],
            patch_size=32,
            patches_per_tile=2,
            batch_size=2,
            num_workers=0,
            class_set=class_set,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture
    def config(self, class_set: int) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", f"chesapeake_cvpr_{class_set}.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def task(
        self, config: Dict[str, Any], monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> ChesapeakeCVPRSegmentationTask:
        task = ChesapeakeCVPRSegmentationTask(**config)
        trainer = FakeTrainer()
        monkeypatch.setattr(task, "trainer", trainer)  # type: ignore[attr-defined]
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_validation(
        self, datamodule: ChesapeakeCVPRDataModule, task: ChesapeakeCVPRSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)
