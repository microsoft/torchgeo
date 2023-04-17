# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, cast

import pytest
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from _pytest.monkeypatch import MonkeyPatch
from lightning.pytorch import LightningDataModule, Trainer
from omegaconf import OmegaConf
from torch.nn.modules import Module

from torchgeo.datamodules import (
    ChesapeakeCVPRDataModule,
    DeepGlobeLandCoverDataModule,
    ETCI2021DataModule,
    GID15DataModule,
    InriaAerialImageLabelingDataModule,
    L7IrishDataModule,
    L8BiomeDataModule,
    LandCoverAIDataModule,
    LoveDADataModule,
    MisconfigurationException,
    NAIPChesapeakeDataModule,
    Potsdam2DDataModule,
    SEN12MSDataModule,
    SpaceNet1DataModule,
    Vaihingen2DDataModule,
)
from torchgeo.datasets import LandCoverAI
from torchgeo.trainers import SemanticSegmentationTask


class SegmentationTestModel(Module):
    def __init__(
        self, in_channels: int = 3, classes: int = 1000, **kwargs: Any
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


def create_model(**kwargs: Any) -> Module:
    return SegmentationTestModel(**kwargs)


def plot(*args: Any, **kwargs: Any) -> None:
    raise ValueError


class TestSemanticSegmentationTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("chesapeake_cvpr_5", ChesapeakeCVPRDataModule),
            ("chesapeake_cvpr_7", ChesapeakeCVPRDataModule),
            ("deepglobelandcover", DeepGlobeLandCoverDataModule),
            ("etci2021", ETCI2021DataModule),
            ("gid15", GID15DataModule),
            ("inria", InriaAerialImageLabelingDataModule),
            ("l7irish", L7IrishDataModule),
            ("l8biome", L8BiomeDataModule),
            ("landcoverai", LandCoverAIDataModule),
            ("loveda", LoveDADataModule),
            ("naipchesapeake", NAIPChesapeakeDataModule),
            ("potsdam2d", Potsdam2DDataModule),
            ("sen12ms_all", SEN12MSDataModule),
            ("sen12ms_s1", SEN12MSDataModule),
            ("sen12ms_s2_all", SEN12MSDataModule),
            ("sen12ms_s2_reduced", SEN12MSDataModule),
            ("spacenet1", SpaceNet1DataModule),
            ("vaihingen2d", Vaihingen2DDataModule),
        ],
    )
    def test_trainer(
        self,
        monkeypatch: MonkeyPatch,
        name: str,
        classname: type[LightningDataModule],
        fast_dev_run: bool,
    ) -> None:
        if name == "naipchesapeake":
            pytest.importorskip("zipfile_deflate64")

        if name == "landcoverai":
            sha256 = "ecec8e871faf1bbd8ca525ca95ddc1c1f5213f40afb94599884bd85f990ebd6b"
            monkeypatch.setattr(LandCoverAI, "sha256", sha256)

        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(dict[Any, dict[Any, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        monkeypatch.setattr(smp, "Unet", create_model)
        monkeypatch.setattr(smp, "DeepLabV3Plus", create_model)
        model_kwargs = conf_dict["module"]
        model = SemanticSegmentationTask(**model_kwargs)

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
