# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, Tuple, cast

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from torchvision import models

from torchgeo.trainers import CycloneDataModule, CycloneSimpleRegressionTask

from .test_utils import mocked_log


class TestCycloneSimpleRegressionTask:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "cyclone.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def datamodule(self) -> CycloneDataModule:
        root = os.path.join("tests", "data", "cyclone")
        seed = 0
        batch_size = 1
        num_workers = 0
        dm = CycloneDataModule(root, seed, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.mark.parametrize("model", [("resnet18", models.resnet.ResNet)])
    def test_model(
        self, default_config: Dict[str, Any], model: Tuple[str, Any]
    ) -> None:
        config_string, config_class = model
        default_config["model"] = config_string
        task = CycloneSimpleRegressionTask(**default_config)
        assert isinstance(task.model, config_class)

    def test_invalid_model(self, default_config: Dict[str, Any]) -> None:
        default_config["model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            CycloneSimpleRegressionTask(**default_config)

    def test_training(
        self,
        default_config: Dict[str, Any],
        datamodule: CycloneDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = CycloneSimpleRegressionTask(**default_config)
        batch = next(iter(datamodule.train_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        out = task.training_step(batch, 0)
        assert isinstance(out, torch.Tensor)

    def test_validation(
        self,
        default_config: Dict[str, Any],
        datamodule: CycloneDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = CycloneSimpleRegressionTask(**default_config)
        batch = next(iter(datamodule.val_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        task.validation_step(batch, 0)

    def test_test(
        self,
        default_config: Dict[str, Any],
        datamodule: CycloneDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = CycloneSimpleRegressionTask(**default_config)
        batch = next(iter(datamodule.test_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        task.test_step(batch, 0)

    def test_configure_optimizers(self, default_config: Dict[str, Any]) -> None:
        task = CycloneSimpleRegressionTask(**default_config)
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out


class TestCycloneDataModule:
    @pytest.fixture
    def datamodule(self) -> CycloneDataModule:
        root = os.path.join("tests", "data", "cyclone")
        seed = 0
        batch_size = 1
        num_workers = 0
        dm = CycloneDataModule(root, seed, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: CycloneDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
