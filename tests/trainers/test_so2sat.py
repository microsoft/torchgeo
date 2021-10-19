# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, Tuple, cast

import pytest
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from torchvision import models

from torchgeo.trainers import So2SatClassificationTask, So2SatDataModule

from .test_utils import mocked_log


class TestSo2SatClassificationTask:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load(os.path.join("conf", "task_defaults", "so2sat.yaml"))
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def datamodule(self) -> So2SatDataModule:
        root = os.path.join("tests", "data", "so2sat")
        batch_size = 1
        num_workers = 0
        bands = "rgb"
        unsupervised_mode = False
        dm = So2SatDataModule(root, batch_size, num_workers, bands, unsupervised_mode)
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.mark.parametrize(
        "classification_model", [("resnet18", models.resnet.ResNet)]
    )
    def test_classification_model(
        self, default_config: Dict[str, Any], classification_model: Tuple[str, Any]
    ) -> None:
        config_string, config_class = classification_model
        default_config["classification_model"] = config_string
        task = So2SatClassificationTask(**default_config)
        assert isinstance(task.model, config_class)

    @pytest.mark.parametrize(
        "loss",
        [
            ("ce", nn.CrossEntropyLoss),  # type: ignore[attr-defined]
            ("jaccard", smp.losses.JaccardLoss),
            ("focal", smp.losses.FocalLoss),
        ],
    )
    def test_loss(self, default_config: Dict[str, Any], loss: Tuple[str, Any]) -> None:
        config_string, config_class = loss
        default_config["loss"] = config_string
        task = So2SatClassificationTask(**default_config)
        assert isinstance(task.loss, config_class)

    @pytest.mark.parametrize("weights", ["imagenet", "random"])
    def test_weights(self, default_config: Dict[str, Any], weights: str) -> None:
        default_config["weights"] = weights
        task = So2SatClassificationTask(**default_config)
        assert task.hparams["weights"] == weights

    def test_configure_optimizers(self, default_config: Dict[str, Any]) -> None:
        task = So2SatClassificationTask(**default_config)
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self,
        default_config: Dict[str, Any],
        datamodule: So2SatDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = So2SatClassificationTask(**default_config)
        batch = next(iter(datamodule.train_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        out = task.training_step(batch, 0)
        assert isinstance(out, torch.Tensor)
        task.training_epoch_end(0)

    def test_validation(
        self,
        default_config: Dict[str, Any],
        datamodule: So2SatDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = So2SatClassificationTask(**default_config)
        batch = next(iter(datamodule.val_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]

        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self,
        default_config: Dict[str, Any],
        datamodule: So2SatDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = So2SatClassificationTask(**default_config)
        batch = next(iter(datamodule.test_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]

        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_invalid_model(self, default_config: Dict[str, Any]) -> None:
        default_config["classification_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            So2SatClassificationTask(**default_config)

    def test_invalid_loss(self, default_config: Dict[str, Any]) -> None:
        default_config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            So2SatClassificationTask(**default_config)

    def test_invalid_weights(self, default_config: Dict[str, Any]) -> None:
        default_config["weights"] = "invalid_weights"
        error_message = "Weight type 'invalid_weights' is not valid."
        with pytest.raises(ValueError, match=error_message):
            So2SatClassificationTask(**default_config)


class TestSo2SatDataModule:
    @pytest.fixture(params=[True, False])
    def unsupervised_mode(self, request: SubRequest) -> bool:
        return cast(bool, request.param)

    @pytest.fixture(params=["rgb", "s2"])
    def bands(self, request: SubRequest) -> bool:
        return cast(bool, request.param)

    @pytest.fixture
    def datamodule(self, bands: str, unsupervised_mode: bool) -> So2SatDataModule:
        root = os.path.join("tests", "data", "so2sat")
        batch_size = 1
        num_workers = 0
        dm = So2SatDataModule(root, batch_size, num_workers, bands, unsupervised_mode)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: So2SatDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: So2SatDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: So2SatDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
