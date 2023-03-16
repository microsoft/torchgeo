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
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from lightning import LightningDataModule, Trainer
from omegaconf import OmegaConf
from torchvision.models import resnet18
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import ChesapeakeCVPRDataModule, MisconfigurationException
from torchgeo.datasets import ChesapeakeCVPR
from torchgeo.models import get_model_weights, list_models
from torchgeo.samplers import GridGeoSampler
from torchgeo.trainers import BYOLTask
from torchgeo.trainers.byol import BYOL, SimCLRAugmentation

from .test_segmentation import SegmentationTestModel


def load(url: str, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    state_dict: Dict[str, Any] = torch.load(url)
    return state_dict


class PredictBYOLDataModule(ChesapeakeCVPRDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = ChesapeakeCVPR(
            splits=self.test_splits, layers=self.layers, **self.kwargs
        )
        self.predict_sampler = GridGeoSampler(
            self.predict_dataset, self.original_patch_size, self.original_patch_size
        )


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
        model = BYOLTask(**model_kwargs)

        model.backbone = SegmentationTestModel(**model_kwargs)

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
            "backbone": "resnet18",
            "in_channels": 13,
            "loss": "ce",
            "num_classes": 10,
            "weights": None,
        }

    @pytest.fixture(
        params=[
            weights for model in list_models() for weights in get_model_weights(model)
        ]
    )
    def weights(self, request: SubRequest) -> WeightsEnum:
        return request.param

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

    def test_weight_file(self, model_kwargs: Dict[str, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        with pytest.warns(UserWarning):
            BYOLTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: Dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = mocked_weights
        BYOLTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: Dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = str(mocked_weights)
        BYOLTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_enum_download(
        self, model_kwargs: Dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = weights
        BYOLTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_str_download(
        self, model_kwargs: Dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = str(weights)
        BYOLTask(**model_kwargs)

    def test_predict(self, model_kwargs: Dict[Any, Any], fast_dev_run: bool) -> None:
        datamodule = PredictBYOLDataModule(
            root="tests/data/chesapeake/cvpr",
            train_splits=["de-test"],
            val_splits=["de-test"],
            test_splits=["de-test"],
            batch_size=1,
            patch_size=64,
            num_workers=0,
        )
        model_kwargs["in_channels"] = 4
        model = BYOLTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=fast_dev_run, log_every_n_steps=1, max_epochs=1)
        trainer.predict(model=model, datamodule=datamodule)
