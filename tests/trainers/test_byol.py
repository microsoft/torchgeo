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
from torchvision.models import resnet18
from torchvision.models._api import WeightsEnum

from torchgeo.datasets import SSL4EOS12, SeasonalContrastS2
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import BYOLTask
from torchgeo.trainers.byol import BYOL, SimCLRAugmentation

from .test_segmentation import SegmentationTestModel


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
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
        "name",
        [
            "chesapeake_cvpr_prior_byol",
            "seco_byol_1",
            "seco_byol_2",
            "ssl4eo_l_byol_1",
            "ssl4eo_l_byol_2",
            "ssl4eo_s12_byol_1",
            "ssl4eo_s12_byol_2",
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))

        if name.startswith("seco"):
            monkeypatch.setattr(SeasonalContrastS2, "__len__", lambda self: 2)

        if name.startswith("ssl4eo_s12"):
            monkeypatch.setattr(SSL4EOS12, "__len__", lambda self: 2)

        # Instantiate datamodule
        datamodule = instantiate(conf.datamodule)

        # Instantiate model
        model = instantiate(conf.module)
        model.backbone = SegmentationTestModel(**conf.module)

        # Instantiate trainer
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model=model, datamodule=datamodule)

    @pytest.fixture
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "backbone": "resnet18",
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
            BYOLTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = mocked_weights
        BYOLTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = str(mocked_weights)
        BYOLTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_enum_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = weights
        BYOLTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_str_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = str(weights)
        BYOLTask(**model_kwargs)
