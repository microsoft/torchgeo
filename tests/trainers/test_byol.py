# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, Dict, Type, cast

import pytest
import timm
import torch
import torch.nn as nn
import torchvision
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer
from torchvision.models import resnet18
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import (
    ChesapeakeCVPRDataModule,
    GeoDataModule,
    NonGeoDataModule,
)
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import BYOLTask
from torchgeo.trainers.byol import BYOL, SimCLRAugmentation

from .test_utils import SegmentationTestModel


def load(url: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    state_dict: Dict[str, Any] = torch.load(url)
    return state_dict


class TestBYOL:
    def test_custom_augment_fn(self) -> None:
        backbone = resnet18()
        layer = backbone.conv1
        new_layer = nn.Conv2d(
            in_channels=4,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias,
        ).requires_grad_()
        backbone.conv1 = new_layer
        augment_fn = SimCLRAugmentation((2, 2))
        BYOL(backbone, augment_fn=augment_fn)


class TestBYOLTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("chesapeake_cvpr_7", ChesapeakeCVPRDataModule),
            ("chesapeake_cvpr_prior", ChesapeakeCVPRDataModule),
        ],
    )
    def test_trainer(self, name: str, classname: Type[LightningDataModule]) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[str, Dict[str, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = BYOLTask(**model_kwargs)

        model.backbone = SegmentationTestModel(**model_kwargs)

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

    @pytest.fixture
    def model_kwargs(self) -> Dict[str, Any]:
        return {"backbone": "resnet18", "weights": None, "in_channels": 3}

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
        BYOLTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: Dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["weights"] = mocked_weights
        BYOLTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: Dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["weights"] = str(mocked_weights)
        BYOLTask(**model_kwargs)
