# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, cast

import pytest
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from omegaconf import OmegaConf

from torchgeo.trainers import SEN12MSDataModule, SEN12MSSegmentationTask


class TestSEN12MSSegmentationTask:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load("conf/task_defaults/sen12ms.yaml")
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    @pytest.fixture
    def datamodule(self) -> None:
        root = os.path.join("tests", "data", "sen12ms")
        seed = 0
        band_set = "all"
        batch_size = 1
        num_workers = 0
        dm = SEN12MSDataModule(root, seed, band_set, batch_size, num_workers)
        dm.prepare_data()
        dm.setup()
        return dm

    def test_unet_ce(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "unet"
        default_config["loss"] = "ce"
        default_config["optimizer"] = "adamw"
        task = SEN12MSSegmentationTask(**default_config)
        assert isinstance(task.model, smp.Unet)
        assert isinstance(task.loss, nn.CrossEntropyLoss)  # type: ignore[attr-defined]

    def test_unet_jaccard(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "unet"
        default_config["loss"] = "jaccard"
        task = SEN12MSSegmentationTask(**default_config)
        assert isinstance(task.model, smp.Unet)
        assert isinstance(task.loss, smp.losses.JaccardLoss)

    def test_configure_optimizers(self, default_config: Dict[str, Any]) -> None:
        task = SEN12MSSegmentationTask(**default_config)
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self, default_config: Dict[str, Any], datamodule: SEN12MSDataModule
    ) -> None:
        task = SEN12MSSegmentationTask(**default_config)
        batch = next(iter(datamodule.train_dataloader()))
        out = task.training_step(batch, 0)
        assert isinstance(out, torch.Tensor)
        task.training_epoch_end(0)

    def test_validation(
        self, default_config: Dict[str, Any], datamodule: SEN12MSDataModule
    ) -> None:
        task = SEN12MSSegmentationTask(**default_config)
        batch = next(iter(datamodule.val_dataloader()))
        out = task.validation_step(batch, 0)
        assert isinstance(out, torch.Tensor)
        task.validation_epoch_end(0)

    def test_test(
        self, default_config: Dict[str, Any], datamodule: SEN12MSDataModule
    ) -> None:
        task = SEN12MSSegmentationTask(**default_config)
        batch = next(iter(datamodule.test_dataloader()))
        out = task.test_step(batch, 0)
        assert isinstance(out, torch.Tensor)
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
    def datamodule(self, request: SubRequest) -> None:
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
