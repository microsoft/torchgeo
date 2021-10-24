# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import os
from typing import Any, Dict, Generator, cast

import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.trainers import LandcoverAIDataModule, LandcoverAISegmentationTask

from .test_utils import FakeTrainer, mocked_log


class TestLandcoverAISegmentationTask:
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "landcoverai.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def datamodule(self) -> LandcoverAIDataModule:
        root = os.path.join("tests", "data", "landcoverai")
        batch_size = 2
        num_workers = 0
        dm = LandcoverAIDataModule(root, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

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
    ) -> LandcoverAISegmentationTask:
        segmentation_model, loss = request.param
        config["segmentation_model"] = segmentation_model
        config["loss"] = loss
        config["verbose"] = True
        task = LandcoverAISegmentationTask(**config)
        trainer = FakeTrainer()
        task.trainer = trainer  # type: ignore[assignment]
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_training(
        self,
        datamodule: LandcoverAIDataModule,
        task: LandcoverAISegmentationTask,
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self,
        datamodule: LandcoverAIDataModule,
        task: LandcoverAISegmentationTask,
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self,
        datamodule: LandcoverAIDataModule,
        task: LandcoverAISegmentationTask,
    ) -> None:
        batch = next(iter(datamodule.test_dataloader()))
        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_configure_optimizers(self, config: Dict[str, Any]) -> None:
        task = LandcoverAISegmentationTask(**config)
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_invalid_model(self, config: Dict[str, Any]) -> None:
        config["segmentation_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            LandcoverAISegmentationTask(**config)

    def test_invalid_loss(self, config: Dict[str, Any]) -> None:
        config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            LandcoverAISegmentationTask(**config)


class TestLandcoverAIDataModule:
    @pytest.fixture
    def datamodule(self) -> LandcoverAIDataModule:
        root = os.path.join("tests", "data", "landcoverai")
        batch_size = 1
        num_workers = 0
        dm = LandcoverAIDataModule(root, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: LandcoverAIDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: LandcoverAIDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: LandcoverAIDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
