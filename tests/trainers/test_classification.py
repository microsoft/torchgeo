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
    GeoDataModule,
    NonGeoDataModule,
    RESISC45DataModule,
    So2SatDataModule,
    UCMercedDataModule,
)
from torchgeo.models import ResNet18_Weights
from torchgeo.datasets import BigEarthNet, EuroSAT
from torchgeo.trainers import ClassificationTask, MultiLabelClassificationTask

from .test_utils import ClassificationTestModel


def create_model(*args: Any, **kwargs: Any) -> Module:
    return ClassificationTestModel(**kwargs)


def load(url: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    state_dict: Dict[str, Any] = torch.load(url)
    return state_dict


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
        self, monkeypatch: MonkeyPatch, name: str, classname: Type[LightningDataModule]
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
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.fit(model=model, datamodule=datamodule)

        if isinstance(datamodule, GeoDataModule):
            if datamodule.test_dataset or datamodule.test_sampler:
                trainer.test(model=model, datamodule=datamodule)
            if datamodule.predict_dataset or datamodule.predict_sampler:
                trainer.predict(model=model, datamodule=datamodule)
        elif isinstance(datamodule, NonGeoDataModule):
            if datamodule.test_dataset or datamodule.dataset:
                trainer.test(model=model, datamodule=datamodule)
            if datamodule.predict_dataset or datamodule.dataset:
                trainer.predict(model=model, datamodule=datamodule)

    def test_no_logger(self) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", "ucmerced.yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[str, Dict[str, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = UCMercedDataModule(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = ClassificationTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(
            logger=False, fast_dev_run=True, log_every_n_steps=1, max_epochs=1
        )
        trainer.fit(model=model, datamodule=datamodule)

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

    def test_missing_attributes(
        self, model_kwargs: Dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delattr(EuroSAT, "plot")
        datamodule = EuroSATDataModule(
            root="tests/data/eurosat", batch_size=1, num_workers=0
        )
        model = ClassificationTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.validate(model=model, datamodule=datamodule)


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
        self, monkeypatch: MonkeyPatch, name: str, classname: Type[LightningDataModule]
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
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
        trainer.predict(model=model, dataloaders=datamodule.val_dataloader())

    def test_no_logger(self) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", "bigearthnet_s1.yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[str, Dict[str, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = BigEarthNetDataModule(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = MultiLabelClassificationTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(
            logger=False, fast_dev_run=True, log_every_n_steps=1, max_epochs=1
        )
        trainer.fit(model=model, datamodule=datamodule)

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

    def test_missing_attributes(
        self, model_kwargs: Dict[str, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delattr(BigEarthNet, "plot")
        datamodule = BigEarthNetDataModule(
            root="tests/data/bigearthnet", batch_size=1, num_workers=0
        )
        model = MultiLabelClassificationTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.validate(model=model, datamodule=datamodule)
