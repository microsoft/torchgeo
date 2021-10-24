# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools
import os
from typing import Any, Dict, Generator, cast

import pytest
import pytorch_lightning as pl
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
            patch_size=128,
            patches_per_tile=2,
            batch_size=2,
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
            [5, 7],
        )
    )
    def task(
        self,
        config: Dict[str, Any],
        request: SubRequest,
        monkeypatch: Generator[MonkeyPatch, None, None],
    ) -> ChesapeakeCVPRSegmentationTask:
        segmentation_model, loss, class_set = request.param
        config["segmentation_model"] = segmentation_model
        config["loss"] = loss
        config["class_set"] = class_set
        task = ChesapeakeCVPRSegmentationTask(**config)
        trainer = pl.Trainer()
        task.trainer = trainer
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
    @pytest.fixture(params=[5, 7])
    def datamodule(self, request: SubRequest) -> ChesapeakeCVPRDataModule:
        dm = ChesapeakeCVPRDataModule(
            os.path.join("tests", "data", "chesapeake", "cvpr"),
            ["de-test"],
            ["de-test"],
            ["de-test"],
            patches_per_tile=2,
            batch_size=2,
            num_workers=0,
            class_set=request.param,
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

    def test_nodata_check(self, datamodule: ChesapeakeCVPRDataModule) -> None:
        nodata_check = datamodule.nodata_check()
        sample = {
            "image": torch.ones(1, 256, 256),
            "mask": torch.ones(256, 256),
        }
        out = nodata_check(sample)
        assert torch.equal(out["image"], torch.zeros(1, 512, 512))
        assert torch.equal(out["mask"], torch.zeros(512, 512))
