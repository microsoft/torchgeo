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

from torchgeo.trainers import SEN12MSDataModule, SEN12MSSegmentationTask


def mocked_log(*args: Any, **kwargs: Any) -> None:
    pass


class TestSEN12MSSegmentationTask:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load("conf/task_defaults/sen12ms.yaml")
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def datamodule(self) -> SEN12MSDataModule:
        root = os.path.join("tests", "data", "sen12ms")
        seed = 0
        band_set = "all"
        batch_size = 1
        num_workers = 0
        dm = SEN12MSDataModule(root, seed, band_set, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.mark.parametrize("segmentation_model", [("unet", smp.Unet)])
    def test_segmentation_model(
        self, default_config: Dict[str, Any], segmentation_model: Tuple[str, Any]
    ) -> None:
        config_string, config_class = segmentation_model
        default_config["segmentation_model"] = config_string
        task = SEN12MSSegmentationTask(**default_config)
        assert isinstance(task.model, config_class)

    @pytest.mark.parametrize(
        "loss",
        [
            ("ce", nn.CrossEntropyLoss),  # type: ignore[attr-defined]
            ("jaccard", smp.losses.JaccardLoss),
        ],
    )
    def test_loss(self, default_config: Dict[str, Any], loss: Tuple[str, Any]) -> None:
        config_string, config_class = loss
        default_config["loss"] = config_string
        task = SEN12MSSegmentationTask(**default_config)
        assert isinstance(task.loss, config_class)

    def test_configure_optimizers(self, default_config: Dict[str, Any]) -> None:
        task = SEN12MSSegmentationTask(**default_config)
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self,
        default_config: Dict[str, Any],
        datamodule: SEN12MSDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = SEN12MSSegmentationTask(**default_config)
        batch = next(iter(datamodule.train_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        out = task.training_step(batch, 0)
        assert isinstance(out, torch.Tensor)
        task.training_epoch_end(0)

    def test_validation(
        self,
        default_config: Dict[str, Any],
        datamodule: SEN12MSDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = SEN12MSSegmentationTask(**default_config)
        batch = next(iter(datamodule.val_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]

        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self,
        default_config: Dict[str, Any],
        datamodule: SEN12MSDataModule,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> None:
        task = SEN12MSSegmentationTask(**default_config)
        batch = next(iter(datamodule.test_dataloader()))
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]

        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_invalid_model(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            SEN12MSSegmentationTask(**default_config)

    def test_invalid_loss(self, default_config: Dict[str, Any]) -> None:
        default_config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            SEN12MSSegmentationTask(**default_config)


class TestSEN12MSDataModule:
    @pytest.fixture(params=["all", "s1", "s2-all", "s2-reduced"])
    def datamodule(self, request: SubRequest) -> SEN12MSDataModule:
        root = os.path.join("tests", "data", "sen12ms")
        seed = 0
        band_set = request.param
        batch_size = 1
        num_workers = 0
        dm = SEN12MSDataModule(root, seed, band_set, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_train_dataloader(self, datamodule: SEN12MSDataModule) -> None:
        next(iter(datamodule.train_dataloader()))

    def test_val_dataloader(self, datamodule: SEN12MSDataModule) -> None:
        next(iter(datamodule.val_dataloader()))

    def test_test_dataloader(self, datamodule: SEN12MSDataModule) -> None:
        next(iter(datamodule.test_dataloader()))
