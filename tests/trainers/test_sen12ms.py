# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, Tuple, cast

import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.datasets import SEN12MSDataModule
from torchgeo.trainers import SEN12MSSegmentationTask

from .test_utils import mocked_log


@pytest.fixture(
    scope="module", params=[("all", 15), ("s1", 2), ("s2-all", 13), ("s2-reduced", 6)]
)
def bands(request: SubRequest) -> Tuple[str, int]:
    return cast(Tuple[str, int], request.param)


@pytest.fixture(scope="module")
def datamodule(bands: Tuple[str, int]) -> SEN12MSDataModule:
    root = os.path.join("tests", "data", "sen12ms")
    seed = 0
    band_set = bands[0]
    batch_size = 1
    num_workers = 0
    dm = SEN12MSDataModule(root, seed, band_set, batch_size, num_workers)
    dm.prepare_data()
    dm.setup()
    return dm


class TestSEN12MSSegmentationTask:
    @pytest.fixture(params=["ce", "jaccard"])
    def config(self, bands: Tuple[str, int], request: SubRequest) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "sen12ms.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        task_args["in_channels"] = bands[1]
        task_args["loss"] = request.param
        return task_args

    @pytest.fixture
    def task(
        self, config: Dict[str, Any], monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> SEN12MSSegmentationTask:
        task = SEN12MSSegmentationTask(**config)
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_configure_optimizers(self, task: SEN12MSSegmentationTask) -> None:
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self, datamodule: SEN12MSDataModule, task: SEN12MSSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self, datamodule: SEN12MSDataModule, task: SEN12MSSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self, datamodule: SEN12MSDataModule, task: SEN12MSSegmentationTask
    ) -> None:
        batch = next(iter(datamodule.test_dataloader()))
        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_invalid_model(self, config: Dict[str, Any]) -> None:
        config["segmentation_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            SEN12MSSegmentationTask(**config)

    def test_invalid_loss(self, config: Dict[str, Any]) -> None:
        config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            SEN12MSSegmentationTask(**config)
