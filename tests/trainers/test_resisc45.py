# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, Tuple, cast

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from torchvision import models

from torchgeo.trainers import RESISC45ClassificationTask, RESISC45DataModule

from .test_utils import mocked_log


class TestRESISC45ClassificationTask:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load("conf/task_defaults/resisc45.yaml")
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

    @pytest.mark.parametrize(
        "classification_model",
        [
            ("resnet18", models.resnet.ResNet),
            ("resnet34", models.resnet.ResNet),
        ],
    )
    def test_classification_model(
        self, default_config: Dict[str, Any], classification_model: Tuple[str, Any]
    ) -> None:
        config_string, config_class = classification_model
        default_config["classification_model"] = config_string
        task = RESISC45ClassificationTask(**default_config)
        assert isinstance(task.model, config_class)

    @pytest.mark.parametrize(
        "loss",
        [
            ("ce", nn.CrossEntropyLoss),  # type: ignore[attr-defined]
        ],
    )
    def test_loss(self, default_config: Dict[str, Any], loss: Tuple[str, Any]) -> None:
        config_string, config_class = loss
        default_config["loss"] = config_string
        task = RESISC45ClassificationTask(**default_config)
        assert isinstance(task.loss, config_class)

    def test_training(
        self,
        default_config: Dict[str, Any],
        datamodule: RESISC45DataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = RESISC45ClassificationTask(**default_config)
        batch = next(iter(datamodule.train_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        out = task.training_step(batch, 0)
        assert isinstance(out, torch.Tensor)
        task.training_epoch_end(0)

    def test_validation(
        self,
        default_config: Dict[str, Any],
        datamodule: RESISC45DataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = RESISC45ClassificationTask(**default_config)
        batch = next(iter(datamodule.val_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]

        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self,
        default_config: Dict[str, Any],
        datamodule: RESISC45DataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = RESISC45ClassificationTask(**default_config)
        batch = next(iter(datamodule.test_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]

        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_invalid_model(self, default_config: Dict[str, Any]) -> None:
        default_config["classification_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            RESISC45ClassificationTask(**default_config)

    def test_invalid_loss(self, default_config: Dict[str, Any]) -> None:
        default_config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            RESISC45ClassificationTask(**default_config)

    def test_invalid_weights(self, default_config: Dict[str, Any]) -> None:
        default_config["weights"] = "invalid_weights"
        error_message = "Weight type 'invalid_weights' is not valid."
        with pytest.raises(ValueError, match=error_message):
            RESISC45ClassificationTask(**default_config)

    def test_configure_optimizers(self, default_config: Dict[str, Any]) -> None:
        task = RESISC45ClassificationTask(**default_config)
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out


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
