# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Type, cast

import pytest
import segmentation_models_pytorch as smp
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer
from torch.nn.modules import Module

from torchgeo.datamodules import (
    ChesapeakeCVPRDataModule,
    ETCI2021DataModule,
    InriaAerialImageLabelingDataModule,
    LandCoverAIDataModule,
    NAIPChesapeakeDataModule,
    OSCDDataModule,
    SEN12MSDataModule,
)
from torchgeo.trainers import SemanticSegmentationTask

from .test_utils import SegmentationTestModel


def create_model(**kwargs: Any) -> Module:
    return SegmentationTestModel(**kwargs)


class TestSemanticSegmentationTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("chesapeake_cvpr_5", ChesapeakeCVPRDataModule),
            ("etci2021", ETCI2021DataModule),
            ("inria", InriaAerialImageLabelingDataModule),
            ("landcoverai", LandCoverAIDataModule),
            ("naipchesapeake", NAIPChesapeakeDataModule),
            ("oscd_all", OSCDDataModule),
            ("oscd_rgb", OSCDDataModule),
            ("sen12ms_all", SEN12MSDataModule),
            ("sen12ms_s1", SEN12MSDataModule),
            ("sen12ms_s2_all", SEN12MSDataModule),
            ("sen12ms_s2_reduced", SEN12MSDataModule),
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, classname: Type[LightningDataModule]
    ) -> None:
        if name == "naipchesapeake":
            pytest.importorskip("zipfile_deflate64")

        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        monkeypatch.setattr(smp, "Unet", create_model)
        monkeypatch.setattr(smp, "DeepLabV3Plus", create_model)
        model_kwargs = conf_dict["module"]
        model = SemanticSegmentationTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1)
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

    def test_no_logger(self) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", "landcoverai.yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = LandCoverAIDataModule(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = SemanticSegmentationTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(logger=False, fast_dev_run=True, log_every_n_steps=1)
        trainer.fit(model=model, datamodule=datamodule)

    @pytest.fixture
    def model_kwargs(self) -> Dict[Any, Any]:
        return {
            "segmentation_model": "unet",
            "encoder_name": "resnet18",
            "encoder_weights": None,
            "in_channels": 1,
            "num_classes": 1,
            "loss": "ce",
            "ignore_zeros": True,
        }

    def test_invalid_model(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["segmentation_model"] = "invalid_model"
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_invalid_loss(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["loss"] = "invalid_loss"
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(**model_kwargs)
