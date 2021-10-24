# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys
from typing import Any, Dict, Generator, cast

import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.trainers import RESISC45ClassificationTask, RESISC45DataModule

from .test_utils import mocked_log


@pytest.mark.skipif(sys.platform == "win32", reason="requires unrar executable")
class TestRESISC45ClassificationTask:
    @pytest.fixture
    def config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "resisc45.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def datamodule(self) -> RESISC45DataModule:
        root = os.path.join("tests", "data", "resisc45")
        batch_size = 2
        num_workers = 0
        dm = RESISC45DataModule(
            root,
            batch_size,
            num_workers,
            val_split_pct=0.33,
            test_split_pct=0.33,
            unsupervised_mode=False,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture(params=["resnet18", "resnet34"])
    def task(
        self,
        config: Dict[str, Any],
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> RESISC45ClassificationTask:
        config["classification_model"] = request.param
        task = RESISC45ClassificationTask(**config)
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_configure_optimizers(self, task: RESISC45ClassificationTask) -> None:
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self, datamodule: RESISC45DataModule, task: RESISC45ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self, datamodule: RESISC45DataModule, task: RESISC45ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self, datamodule: RESISC45DataModule, task: RESISC45ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.test_dataloader()))
        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_invalid_model(self, config: Dict[str, Any]) -> None:
        config["classification_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            RESISC45ClassificationTask(**config)

    def test_invalid_loss(self, config: Dict[str, Any]) -> None:
        config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            RESISC45ClassificationTask(**config)

    def test_invalid_weights(self, config: Dict[str, Any]) -> None:
        config["weights"] = "invalid_weights"
        error_message = "Weight type 'invalid_weights' is not valid."
        with pytest.raises(ValueError, match=error_message):
            RESISC45ClassificationTask(**config)


@pytest.mark.skipif(sys.platform == "win32", reason="requires unrar executable")
class TestRESISC45DataModule:
    @pytest.fixture(params=[True, False])  # Fixture for unsupervised mode
    def datamodule(self, request: SubRequest) -> RESISC45DataModule:
        root = os.path.join("tests", "data", "resisc45")
        batch_size = 2
        num_workers = 0
        unsupervised_mode = request.param
        dm = RESISC45DataModule(
            root,
            batch_size,
            num_workers,
            val_split_pct=0.33,
            test_split_pct=0.33,
            unsupervised_mode=unsupervised_mode,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: RESISC45DataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: RESISC45DataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: RESISC45DataModule) -> None:
        next(iter(datamodule.test_dataloader()))
