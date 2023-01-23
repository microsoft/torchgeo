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
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import (
    COWCCountingDataModule,
    MisconfigurationException,
    TropicalCycloneDataModule,
)
from torchgeo.datasets import TropicalCyclone
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import RegressionTask

from .test_utils import RegressionTestModel


def load(url: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    state_dict: Dict[str, Any] = torch.load(url)
    return state_dict


class PredictRegressionDataModule(TropicalCycloneDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = TropicalCyclone(split="test", **self.kwargs)


def plot(*args: Any, **kwargs: Any) -> None:
    raise ValueError


class TestRegressionTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("cowc_counting", COWCCountingDataModule),
            ("cyclone", TropicalCycloneDataModule),
        ],
    )
    def test_trainer(
        self, name: str, classname: Type[LightningDataModule], fast_dev_run: bool
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[str, Dict[str, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = RegressionTask(**model_kwargs)

        model.model = RegressionTestModel()

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
            "weights": None,
            "num_outputs": 1,
            "in_channels": 3,
        }

    @pytest.fixture
    def mocked_weights(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> WeightsEnum:
        weights = ResNet18_Weights.SENTINEL2_RGB_MOCO
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model("resnet18", in_chans=weights.meta["in_chans"])
        torch.save(model.state_dict(), path)
        monkeypatch.setattr(weights, "url", str(path))
        monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", load)
        return weights

    def test_weight_file(self, model_kwargs: Dict[str, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        with pytest.warns(UserWarning):
            RegressionTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: Dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["weights"] = mocked_weights
        with pytest.warns(UserWarning):
            RegressionTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: Dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["weights"] = str(mocked_weights)
        with pytest.warns(UserWarning):
            RegressionTask(**model_kwargs)

    def test_no_rgb(
        self, monkeypatch: MonkeyPatch, model_kwargs: Dict[Any, Any], fast_dev_run: bool
    ) -> None:
        monkeypatch.setattr(TropicalCycloneDataModule, "plot", plot)
        datamodule = TropicalCycloneDataModule(
            root="tests/data/cyclone", batch_size=1, num_workers=0
        )
        model = RegressionTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, model_kwargs: Dict[Any, Any], fast_dev_run: bool) -> None:
        datamodule = PredictRegressionDataModule(
            root="tests/data/cyclone", batch_size=1, num_workers=0
        )
        model = RegressionTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
        trainer.predict(model=model, datamodule=datamodule)
