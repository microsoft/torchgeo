# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, Dict, Type, cast

import pytest
import timm
import torch
import torchvision
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import (
    BigEarthNetDataModule,
    EuroSATDataModule,
    MisconfigurationException,
    RESISC45DataModule,
    So2SatDataModule,
    UCMercedDataModule,
)
from torchgeo.datasets import BigEarthNet, EuroSAT
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask, MultiLabelClassificationTask

from .test_utils import ClassificationTestModel


class PredictClassificationDataModule(EuroSATDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = EuroSAT(split="test", **self.kwargs)


class PredictMultiLabelClassificationDataModule(BigEarthNetDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = BigEarthNet(split="test", **self.kwargs)


def create_model(*args: Any, **kwargs: Any) -> Module:
    return ClassificationTestModel(**kwargs)


def load(url: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    state_dict: Dict[str, Any] = torch.load(url)
    return state_dict


def plot(*args: Any, **kwargs: Any) -> None:
    raise ValueError


class TestClassificationTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("eurosat", EuroSATDataModule),
            ("resisc45", RESISC45DataModule),
            ("so2sat_all", So2SatDataModule),
            ("so2sat_s1", So2SatDataModule),
            ("so2sat_s2", So2SatDataModule),
            ("ucmerced", UCMercedDataModule),
        ],
    )
    def test_trainer(
        self,
        monkeypatch: MonkeyPatch,
        name: str,
        classname: Type[LightningDataModule],
        fast_dev_run: bool,
    ) -> None:
        if name.startswith("so2sat"):
            pytest.importorskip("h5py", minversion="2.6")

        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[str, Dict[str, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        monkeypatch.setattr(timm, "create_model", create_model)
        model_kwargs = conf_dict["module"]
        model = ClassificationTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
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
    def model_kwargs(self) -> Dict[str, Any]:
        return {
            "model": "resnet18",
            "in_channels": 13,
            "loss": "ce",
            "num_classes": 10,
            "weights": None,
        }

    @pytest.fixture
    def mocked_weights(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> WeightsEnum:
        weights = ResNet18_Weights.SENTINEL2_ALL_MOCO
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"])
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights, "url", str(path))
        monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", load)
        return weights

    def test_weight_file(self, model_kwargs: Dict[str, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        with pytest.warns(UserWarning):
            ClassificationTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: Dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["weights"] = mocked_weights
        with pytest.warns(UserWarning):
            ClassificationTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: Dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["weights"] = str(mocked_weights)
        with pytest.warns(UserWarning):
            ClassificationTask(**model_kwargs)

    def test_invalid_loss(self, model_kwargs: Dict[str, Any]) -> None:
        model_kwargs["loss"] = "invalid_loss"
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            ClassificationTask(**model_kwargs)

    def test_no_rgb(
        self, monkeypatch: MonkeyPatch, model_kwargs: Dict[Any, Any], fast_dev_run: bool
    ) -> None:
        monkeypatch.setattr(EuroSATDataModule, "plot", plot)
        datamodule = EuroSATDataModule(
            root="tests/data/eurosat", batch_size=1, num_workers=0
        )
        model = ClassificationTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, model_kwargs: Dict[Any, Any], fast_dev_run: bool) -> None:
        datamodule = PredictClassificationDataModule(
            root="tests/data/eurosat", batch_size=1, num_workers=0
        )
        model = ClassificationTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
        trainer.predict(model=model, datamodule=datamodule)


class TestMultiLabelClassificationTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("bigearthnet_all", BigEarthNetDataModule),
            ("bigearthnet_s1", BigEarthNetDataModule),
            ("bigearthnet_s2", BigEarthNetDataModule),
        ],
    )
    def test_trainer(
        self,
        monkeypatch: MonkeyPatch,
        name: str,
        classname: Type[LightningDataModule],
        fast_dev_run: bool,
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[str, Dict[str, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        monkeypatch.setattr(timm, "create_model", create_model)
        model_kwargs = conf_dict["module"]
        model = MultiLabelClassificationTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
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
    def model_kwargs(self) -> Dict[str, Any]:
        return {
            "model": "resnet18",
            "in_channels": 14,
            "loss": "bce",
            "num_classes": 19,
            "weights": None,
        }

    def test_invalid_loss(self, model_kwargs: Dict[str, Any]) -> None:
        model_kwargs["loss"] = "invalid_loss"
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            MultiLabelClassificationTask(**model_kwargs)

    def test_no_rgb(
        self, monkeypatch: MonkeyPatch, model_kwargs: Dict[Any, Any], fast_dev_run: bool
    ) -> None:
        monkeypatch.setattr(BigEarthNetDataModule, "plot", plot)
        datamodule = BigEarthNetDataModule(
            root="tests/data/bigearthnet", batch_size=1, num_workers=0
        )
        model = MultiLabelClassificationTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, model_kwargs: Dict[Any, Any], fast_dev_run: bool) -> None:
        datamodule = PredictMultiLabelClassificationDataModule(
            root="tests/data/bigearthnet", batch_size=1, num_workers=0
        )
        model = MultiLabelClassificationTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
        trainer.predict(model=model, datamodule=datamodule)
