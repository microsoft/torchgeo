# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import os
from typing import Any, Dict, Generator, cast

import pytest
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.trainers import ChesapeakeCVPRDataModule, ChesapeakeCVPRSegmentationTask

from .test_utils import mocked_log


class TestChesapeakeCVPRSegmentationTask:
    @pytest.fixture
    def datamodule(self) -> ChesapeakeCVPRDataModule:
        dm = ChesapeakeCVPRDataModule(
            os.path.join("tests", "data", "chesapeake", "cvpr"),
            ["de-test"],
            ["de-test"],
            ["de-test"],
            patches_per_tile=1,
            batch_size=1,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "chesapeake_cvpr.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture(
        params=itertools.product(
            ["unet", "deeplabv3+", "fcn"],
            ["ce", "jaccard", "focal"],
        )
    )
    def task(
        self,
        config: Dict[str, Any],
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> ChesapeakeCVPRSegmentationTask:
        model, loss = request.param
        config["segmentation_model"] = model
        config["loss"] = loss
        task = ChesapeakeCVPRSegmentationTask(**config)
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_configure_optimizers(self, task: ChesapeakeCVPRSegmentationTask) -> None:
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self,
        datamodule: ChesapeakeCVPRDataModule,
        task: ChesapeakeCVPRSegmentationTask,
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self,
        datamodule: ChesapeakeCVPRDataModule,
        task: ChesapeakeCVPRSegmentationTask,
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self,
        datamodule: ChesapeakeCVPRDataModule,
        task: ChesapeakeCVPRSegmentationTask,
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


class TestChesapeakeCVPRDataModule:
    @pytest.fixture
    def datamodule(self) -> ChesapeakeCVPRDataModule:
        dm = ChesapeakeCVPRDataModule(
            os.path.join("tests", "data", "chesapeake", "cvpr"),
            ["de-test"],
            ["de-test"],
            ["de-test"],
            patches_per_tile=1,
            batch_size=1,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
