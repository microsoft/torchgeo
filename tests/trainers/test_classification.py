# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import (
    BigEarthNetDataModule,
    EuroSATDataModule,
    MisconfigurationException,
)
from torchgeo.datasets import BigEarthNet, EuroSAT
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask, MultiLabelClassificationTask


class ClassificationTestModel(Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 10, **kwargs: Any) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=1, kernel_size=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1, num_classes) if num_classes else nn.Identity()
        self.num_features = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class PredictClassificationDataModule(EuroSATDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = EuroSAT(split="test", **self.kwargs)


class PredictMultiLabelClassificationDataModule(BigEarthNetDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = BigEarthNet(split="test", **self.kwargs)


def create_model(*args: Any, **kwargs: Any) -> Module:
    return ClassificationTestModel(**kwargs)


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


def plot(*args: Any, **kwargs: Any) -> None:
    raise ValueError


class TestClassificationTask:
    @pytest.mark.parametrize(
        "name",
        [
            "eurosat",
            "eurosat100",
            "fire_risk",
            "resisc45",
            "so2sat_all",
            "so2sat_s1",
            "so2sat_s2",
            "so2sat_rgb",
            "ucmerced",
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        if name.startswith("so2sat"):
            pytest.importorskip("h5py", minversion="3")

        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))

        # Instantiate datamodule
        datamodule = instantiate(conf.datamodule)

        # Instantiate model
        monkeypatch.setattr(timm, "create_model", create_model)
        model = instantiate(conf.module)

        # Instantiate trainer
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model=model, datamodule=datamodule)
        try:
            trainer.test(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass
        try:
            trainer.predict(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass

    @pytest.fixture
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "model": "resnet18",
            "in_channels": 13,
            "loss": "ce",
            "num_classes": 10,
            "weights": None,
        }

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ResNet18_Weights.SENTINEL2_ALL_MOCO

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model(
            weights.meta["model"], in_chans=weights.meta["in_chans"]
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, "url", str(path))
        except AttributeError:
            monkeypatch.setattr(weights, "url", str(path))
        monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", load)
        return weights

    def test_weight_file(self, model_kwargs: dict[str, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        with pytest.warns(UserWarning):
            ClassificationTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["model"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = mocked_weights
        with pytest.warns(UserWarning):
            ClassificationTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["model"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = str(mocked_weights)
        with pytest.warns(UserWarning):
            ClassificationTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_enum_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["model"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = weights
        ClassificationTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_str_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["model"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = str(weights)
        ClassificationTask(**model_kwargs)

    def test_invalid_loss(self, model_kwargs: dict[str, Any]) -> None:
        model_kwargs["loss"] = "invalid_loss"
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            ClassificationTask(**model_kwargs)

    def test_no_rgb(
        self, monkeypatch: MonkeyPatch, model_kwargs: dict[Any, Any], fast_dev_run: bool
    ) -> None:
        monkeypatch.setattr(EuroSATDataModule, "plot", plot)
        datamodule = EuroSATDataModule(
            root="tests/data/eurosat", batch_size=1, num_workers=0
        )
        model = ClassificationTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, model_kwargs: dict[Any, Any], fast_dev_run: bool) -> None:
        datamodule = PredictClassificationDataModule(
            root="tests/data/eurosat", batch_size=1, num_workers=0
        )
        model = ClassificationTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    @pytest.mark.parametrize(
        "model_name", ["resnet18", "efficientnetv2_s", "vit_base_patch16_384"]
    )
    def test_freeze_backbone(
        self, model_name: str, model_kwargs: dict[Any, Any]
    ) -> None:
        model_kwargs["freeze_backbone"] = True
        model_kwargs["model"] = model_name
        model = ClassificationTask(**model_kwargs)
        assert not all([param.requires_grad for param in model.model.parameters()])
        assert all(
            [param.requires_grad for param in model.model.get_classifier().parameters()]
        )


class TestMultiLabelClassificationTask:
    @pytest.mark.parametrize(
        "name", ["bigearthnet_all", "bigearthnet_s1", "bigearthnet_s2"]
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))

        # Instantiate datamodule
        datamodule = instantiate(conf.datamodule)

        # Instantiate model
        monkeypatch.setattr(timm, "create_model", create_model)
        model = instantiate(conf.module)

        # Instantiate trainer
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model=model, datamodule=datamodule)
        try:
            trainer.test(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass
        try:
            trainer.predict(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass

    @pytest.fixture
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "model": "resnet18",
            "in_channels": 14,
            "loss": "bce",
            "num_classes": 19,
            "weights": None,
        }

    def test_invalid_loss(self, model_kwargs: dict[str, Any]) -> None:
        model_kwargs["loss"] = "invalid_loss"
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            MultiLabelClassificationTask(**model_kwargs)

    def test_no_rgb(
        self, monkeypatch: MonkeyPatch, model_kwargs: dict[Any, Any], fast_dev_run: bool
    ) -> None:
        monkeypatch.setattr(BigEarthNetDataModule, "plot", plot)
        datamodule = BigEarthNetDataModule(
            root="tests/data/bigearthnet", batch_size=1, num_workers=0
        )
        model = MultiLabelClassificationTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, model_kwargs: dict[Any, Any], fast_dev_run: bool) -> None:
        datamodule = PredictMultiLabelClassificationDataModule(
            root="tests/data/bigearthnet", batch_size=1, num_workers=0
        )
        model = MultiLabelClassificationTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)
