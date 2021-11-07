# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Generator, Optional, cast

import pytest
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset

from torchgeo.trainers import ClassificationTask, MultiLabelClassificationTask

from .test_utils import mocked_log


class DummyDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self, num_channels: int, num_classes: int, multilabel: bool) -> None:
        x = torch.randn(10, num_channels, 128, 128)  # (b, c, h, w)
        y = torch.randint(  # type: ignore[attr-defined]
            0, num_classes, size=(10,)
        )  # (b,)

        if multilabel:
            y = F.one_hot(y, num_classes=num_classes)  # (b, classes)

        self.dataset = TensorDataset(x, y)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Tensor]:
        x, y = self.dataset[idx]
        sample = {"image": x, "label": y}
        return sample


class DummyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        num_channels: int,
        num_classes: int,
        multilabel: bool,
        batch_size: int = 1,
        num_workers: int = 0,
    ) -> None:
        super().__init__()  # type: ignore[no-untyped-call]
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.multilabel = multilabel
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = DummyDataset(
            num_channels=self.num_channels,
            num_classes=self.num_classes,
            multilabel=self.multilabel,
        )

    def train_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self) -> DataLoader:  # type: ignore[type-arg]
        return DataLoader(
            self.dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


class TestClassificationTask:

    num_classes = 10

    @pytest.fixture(scope="class", params=[2, 3, 5])
    def datamodule(self, request: SubRequest) -> DummyDataModule:
        dm = DummyDataModule(
            num_channels=request.param,
            num_classes=self.num_classes,
            multilabel=False,
            batch_size=2,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture(
        scope="class",
        params=zip(
            ["ce", "jaccard", "focal"],
            ["imagenet", "random", "random"],
            ["resnet18", "hrnet_w18_small_v2", "tf_efficientnet_b0"],
        ),
    )
    def config(
        self, request: SubRequest, datamodule: DummyDataModule
    ) -> Dict[str, Any]:
        loss, weights, model = request.param
        task_args: Dict[str, Any] = {}
        task_args["classification_model"] = model
        task_args["learning_rate"] = 3e-4
        task_args["learning_rate_schedule_patience"] = 6
        task_args["in_channels"] = datamodule.num_channels
        task_args["loss"] = loss
        task_args["num_classes"] = self.num_classes
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
        self, datamodule: DummyDataModule, task: ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self, datamodule: DummyDataModule, task: ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(self, datamodule: DummyDataModule, task: ClassificationTask) -> None:
        batch = next(iter(datamodule.test_dataloader()))
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
        error_message = "Model type 'invalid_model' is not a valid timm model."
        with pytest.raises(ValueError, match=error_message):
            ClassificationTask(**config)

    def test_invalid_loss(self, config: Dict[str, Any]) -> None:
        config["loss"] = "invalid_loss"
        config["classification_model"] = "resnet18"
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

    num_classes = 10

    @pytest.fixture(scope="class")
    def datamodule(self, request: SubRequest) -> DummyDataModule:
        dm = DummyDataModule(
            num_channels=3,
            num_classes=self.num_classes,
            multilabel=True,
            batch_size=2,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()
        return dm

    @pytest.fixture(scope="class", params=zip(["bce", "bce"], ["imagenet", "random"]))
    def config(
        self, datamodule: DummyDataModule, request: SubRequest
    ) -> Dict[str, Any]:
        task_args: Dict[str, Any] = {}
        task_args["classification_model"] = "resnet18"
        task_args["learning_rate"] = 3e-4
        task_args["learning_rate_schedule_patience"] = 6
        task_args["in_channels"] = datamodule.num_channels
        loss, weights = request.param
        task_args["loss"] = loss
        task_args["num_classes"] = self.num_classes
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
        self, datamodule: DummyDataModule, task: ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        task.training_step(batch, 0)
        task.training_epoch_end(0)

    def test_validation(
        self, datamodule: DummyDataModule, task: ClassificationTask
    ) -> None:
        batch = next(iter(datamodule.val_dataloader()))
        task.validation_step(batch, 0)
        task.validation_epoch_end(0)

    def test_test(self, datamodule: DummyDataModule, task: ClassificationTask) -> None:
        batch = next(iter(datamodule.test_dataloader()))
        task.test_step(batch, 0)
        task.test_epoch_end(0)

    def test_invalid_loss(self, config: Dict[str, Any]) -> None:
        config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            MultiLabelClassificationTask(**config)
