# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, cast

import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.datasets import ChesapeakeCVPRDataModule
from torchgeo.trainers import ChesapeakeCVPRSegmentationTask

from .test_utils import FakeTrainer, mocked_log


@pytest.fixture(scope="module", params=[5, 7])
def class_set(request: SubRequest) -> int:
    return cast(int, request.param)


@pytest.fixture(scope="module")
def datamodule(class_set: int) -> ChesapeakeCVPRDataModule:
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


class TestChesapeakeCVPRSegmentationTask:
    @pytest.fixture(
        params=zip(["unet", "deeplabv3+", "fcn"], ["ce", "jaccard", "focal"])
    )
    def config(self, class_set: int, request: SubRequest) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "chesapeake_cvpr.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        segmentation_model, loss = request.param
        task_args["class_set"] = class_set
        task_args["segmentation_model"] = segmentation_model
        task_args["loss"] = loss
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

    def test_configure_optimizers(self, task: ChesapeakeCVPRSegmentationTask) -> None:
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self, datamodule: ChesapeakeCVPRDataModule, task: ChesapeakeCVPRSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self, datamodule: ChesapeakeCVPRDataModule, task: ChesapeakeCVPRSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self, datamodule: ChesapeakeCVPRDataModule, task: ChesapeakeCVPRSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.test_dataloader()))
        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_invalid_class_set(self, config: Dict[str, Any]) -> None:
        config["class_set"] = 6
        error_message = "'class_set' must be either 5 or 7"
        with pytest.raises(ValueError, match=error_message):
            ChesapeakeCVPRSegmentationTask(**config)

    def test_invalid_model(self, config: Dict[str, Any]) -> None:
        config["segmentation_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            ChesapeakeCVPRSegmentationTask(**config)

    def test_invalid_loss(self, config: Dict[str, Any]) -> None:
        config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            ChesapeakeCVPRSegmentationTask(**config)
