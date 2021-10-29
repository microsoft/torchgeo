# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, Tuple, cast

import pytest
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf

from torchgeo.trainers import (
    BigEarthNetDataModule,
    ClassificationTask,
    MultiLabelClassificationTask,
    So2SatDataModule,
)

from .test_utils import mocked_log


@pytest.fixture(scope="module", params=[("rgb", 3), ("s2", 10)])
def bands_so2sat(request: SubRequest) -> Tuple[str, int]:
    return cast(Tuple[str, int], request.param)


@pytest.fixture(scope="module", params=[("s1", 2), ("s2", 12), ("all", 14)])
def bands_bigearthnet(request: SubRequest) -> Tuple[str, int]:
    return cast(Tuple[str, int], request.param)


@pytest.fixture(scope="module", params=[True, False])
def datamodule_classification(
    bands_so2sat: Tuple[str, int], request: SubRequest
) -> So2SatDataModule:
    band_set = bands_so2sat[0]
    unsupervised_mode = request.param
    root = os.path.join("tests", "data", "so2sat")
    batch_size = 2
    num_workers = 0
    dm = So2SatDataModule(root, batch_size, num_workers, band_set, unsupervised_mode)
    dm.prepare_data()
    dm.setup()
    return dm


@pytest.fixture(scope="module", params=[True, False])
def datamodule_multilabel(
    bands_bigearthnet: Tuple[str, int], request: SubRequest
) -> BigEarthNetDataModule:
    band_set = bands_bigearthnet[0]
    unsupervised_mode = request.param
    root = os.path.join("tests", "data", "bigearthnet")
    batch_size = 1
    num_workers = 0
    dm = BigEarthNetDataModule(
        root,
        band_set,
        batch_size,
        num_workers,
        unsupervised_mode,
        val_split_pct=0.3,
        test_split_pct=0.3,
    )
    dm.prepare_data()
    dm.setup()
    return dm


class TestClassificationTask:
    @pytest.fixture(
        params=zip(["ce", "jaccard", "focal"], ["imagenet", "random", "random"])
    )
    def config(
        self, request: SubRequest, bands_so2sat: Tuple[str, int]
    ) -> Dict[str, Any]:
        task_conf = OmegaConf.load(os.path.join("conf", "task_defaults", "so2sat.yaml"))
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        task_args["in_channels"] = bands_so2sat[1]
        loss, weights = request.param
        task_args["loss"] = loss
        task_args["weights"] = weights
        return task_args

    @pytest.fixture
    def task(
        self, config: Dict[str, Any], monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> ClassificationTask:
        task = ClassificationTask(**config)
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_configure_optimizers(self, task: ClassificationTask) -> None:
        out = task.configure_optimizers()
        assert "optimizer" in out
        assert "lr_scheduler" in out

    def test_training(
        self, datamodule_classification: So2SatDataModule, task: ClassificationTask
    ) -> None:
        batch = next(iter(datamodule_classification.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self, datamodule_classification: So2SatDataModule, task: ClassificationTask
    ) -> None:
        batch = next(iter(datamodule_classification.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self, datamodule_classification: So2SatDataModule, task: ClassificationTask
    ) -> None:
        batch = next(iter(datamodule_classification.test_dataloader()))
        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_pretrained(self, checkpoint: str) -> None:
        task_conf = OmegaConf.load(os.path.join("conf", "task_defaults", "so2sat.yaml"))
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        task_args["weights"] = checkpoint
        with pytest.warns(UserWarning):
            ClassificationTask(**task_args)

    def test_invalid_model(self, config: Dict[str, Any]) -> None:
        config["classification_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            ClassificationTask(**config)

    def test_invalid_loss(self, config: Dict[str, Any]) -> None:
        config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            ClassificationTask(**config)

    def test_invalid_weights(self, config: Dict[str, Any]) -> None:
        config["weights"] = "invalid_weights"
        error_message = "Weight type 'invalid_weights' is not valid."
        with pytest.raises(ValueError, match=error_message):
            ClassificationTask(**config)

    def test_invalid_pretrained(self, checkpoint: str, config: Dict[str, Any]) -> None:
        config["weights"] = checkpoint
        config["classification_model"] = "resnet50"
        error_message = "Trying to load resnet18 weights into a resnet50"
        with pytest.raises(ValueError, match=error_message):
            ClassificationTask(**config)


class TestMultiLabelClassificationTask:
    @pytest.fixture(params=zip(["bce", "bce"], ["imagenet", "random"]))
    def config(
        self, request: SubRequest, bands_bigearthnet: Tuple[str, int]
    ) -> Dict[str, Any]:
        task_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", "bigearthnet.yaml")
        )
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        task_args["in_channels"] = bands_bigearthnet[1]
        loss, weights = request.param
        task_args["loss"] = loss
        task_args["weights"] = weights
        return task_args

    @pytest.fixture
    def task(
        self, config: Dict[str, Any], monkeypatch: Generator[MonkeyPatch, None, None]
    ) -> MultiLabelClassificationTask:
        task = MultiLabelClassificationTask(**config)
        monkeypatch.setattr(task, "log", mocked_log)  # type: ignore[attr-defined]
        return task

    def test_training(
        self,
        datamodule_multilabel: BigEarthNetDataModule,
        task: MultiLabelClassificationTask,
    ) -> None:
        batch = next(iter(datamodule_multilabel.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self,
        datamodule_multilabel: BigEarthNetDataModule,
        task: MultiLabelClassificationTask,
    ) -> None:
        batch = next(iter(datamodule_multilabel.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(
        self,
        datamodule_multilabel: BigEarthNetDataModule,
        task: MultiLabelClassificationTask,
    ) -> None:
        batch = next(iter(datamodule_multilabel.test_dataloader()))
        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_invalid_loss(self, config: Dict[str, Any]) -> None:
        config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            MultiLabelClassificationTask(**config)
