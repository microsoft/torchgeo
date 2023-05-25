# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, cast

import pytest
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torchvision
from _pytest.fixtures import SubRequest
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import MisconfigurationException, SEN12MSDataModule
from torchgeo.datasets import LandCoverAI
from torchgeo.models import get_model_weights, list_models
from torchgeo.trainers import SemanticSegmentationTask


class SegmentationTestModel(Module):
    def __init__(self, in_channels: int = 3, classes: int = 3, **kwargs: Any) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


def create_model(**kwargs: Any) -> Module:
    return SegmentationTestModel(**kwargs)


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


def plot(*args: Any, **kwargs: Any) -> None:
    raise ValueError


class TestSemanticSegmentationTask:
    @pytest.mark.parametrize(
        "name",
        [
            "chesapeake_cvpr_5",
            "chesapeake_cvpr_7",
            "deepglobelandcover",
            "etci2021",
            "gid15",
            "inria",
            "l7irish",
            "l8biome",
            "landcoverai",
            "loveda",
            "naipchesapeake",
            "potsdam2d",
            "sen12ms_all",
            "sen12ms_s1",
            "sen12ms_s2_all",
            "sen12ms_s2_reduced",
            "spacenet1",
            "ssl4eo_l_benchmark_cdl",
            "ssl4eo_l_benchmark_nlcd",
            "vaihingen2d",
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        if name == "naipchesapeake":
            pytest.importorskip("zipfile_deflate64")

        if name == "landcoverai":
            sha256 = "ecec8e871faf1bbd8ca525ca95ddc1c1f5213f40afb94599884bd85f990ebd6b"
            monkeypatch.setattr(LandCoverAI, "sha256", sha256)

        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))

        # Instantiate datamodule
        datamodule = instantiate(conf.datamodule)

        # Instantiate model
        monkeypatch.setattr(smp, "Unet", create_model)
        monkeypatch.setattr(smp, "DeepLabV3Plus", create_model)
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
    def model_kwargs(self) -> dict[Any, Any]:
        return {
            "model": "unet",
            "backbone": "resnet18",
            "weights": None,
            "in_channels": 3,
            "num_classes": 6,
            "loss": "ce",
            "ignore_index": 0,
        }

    @pytest.fixture(
        params=[
            weights
            for model in list_models()
            for weights in get_model_weights(model)
            if "resnet" in weights.meta["model"]
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

    def test_weight_file(self, model_kwargs: dict[str, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        SemanticSegmentationTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = mocked_weights
        SemanticSegmentationTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = str(mocked_weights)
        SemanticSegmentationTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_enum_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = weights
        SemanticSegmentationTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_str_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = str(weights)
        SemanticSegmentationTask(**model_kwargs)

    def test_invalid_model(self, model_kwargs: dict[Any, Any]) -> None:
        model_kwargs["model"] = "invalid_model"
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_invalid_loss(self, model_kwargs: dict[Any, Any]) -> None:
        model_kwargs["loss"] = "invalid_loss"
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_invalid_ignoreindex(self, model_kwargs: dict[Any, Any]) -> None:
        model_kwargs["ignore_index"] = "0"
        match = "ignore_index must be an int or None"
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_ignoreindex_with_jaccard(self, model_kwargs: dict[Any, Any]) -> None:
        model_kwargs["loss"] = "jaccard"
        model_kwargs["ignore_index"] = 0
        match = "ignore_index has no effect on training when loss='jaccard'"
        with pytest.warns(UserWarning, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_no_rgb(
        self, monkeypatch: MonkeyPatch, model_kwargs: dict[Any, Any], fast_dev_run: bool
    ) -> None:
        model_kwargs["in_channels"] = 15
        monkeypatch.setattr(SEN12MSDataModule, "plot", plot)
        datamodule = SEN12MSDataModule(
            root="tests/data/sen12ms", batch_size=1, num_workers=0
        )
        model = SemanticSegmentationTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    @pytest.mark.parametrize(
        "backbone", ["resnet18", "mobilenet_v2", "efficientnet-b0"]
    )
    @pytest.mark.parametrize("model_name", ["unet", "deeplabv3+"])
    def test_freeze_backbone(
        self, backbone: str, model_name: str, model_kwargs: dict[Any, Any]
    ) -> None:
        model_kwargs["freeze_backbone"] = True
        model_kwargs["model"] = model_name
        model_kwargs["backbone"] = backbone
        model = SemanticSegmentationTask(**model_kwargs)
        assert all(
            [param.requires_grad is False for param in model.model.encoder.parameters()]
        )
        assert all([param.requires_grad for param in model.model.decoder.parameters()])
        assert all(
            [
                param.requires_grad
                for param in model.model.segmentation_head.parameters()
            ]
        )

    @pytest.mark.parametrize("model_name", ["unet", "deeplabv3+"])
    def test_freeze_decoder(
        self, model_name: str, model_kwargs: dict[Any, Any]
    ) -> None:
        model_kwargs["freeze_decoder"] = True
        model_kwargs["model"] = model_name
        model = SemanticSegmentationTask(**model_kwargs)
        assert all(
            [param.requires_grad is False for param in model.model.decoder.parameters()]
        )
        assert all([param.requires_grad for param in model.model.encoder.parameters()])
        assert all(
            [
                param.requires_grad
                for param in model.model.segmentation_head.parameters()
            ]
        )
