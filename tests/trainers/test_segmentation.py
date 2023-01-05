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
    DeepGlobeLandCoverDataModule,
    ETCI2021DataModule,
    GeoDataModule,
    GID15DataModule,
    InriaAerialImageLabelingDataModule,
    LandCoverAIDataModule,
    LoveDADataModule,
    NAIPChesapeakeDataModule,
    NonGeoDataModule,
    Potsdam2DDataModule,
    SEN12MSDataModule,
    SpaceNet1DataModule,
    Vaihingen2DDataModule,
)
from torchgeo.datasets import LandCoverAI
from torchgeo.trainers import SemanticSegmentationTask

from .test_utils import SegmentationTestModel


def create_model(**kwargs: Any) -> Module:
    return SegmentationTestModel(**kwargs)


class TestSemanticSegmentationTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("chesapeake_cvpr_5", ChesapeakeCVPRDataModule),
            ("deepglobelandcover", DeepGlobeLandCoverDataModule),
            ("etci2021", ETCI2021DataModule),
            ("gid15", GID15DataModule),
            ("inria", InriaAerialImageLabelingDataModule),
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
        self, monkeypatch: MonkeyPatch, name: str, classname: Type[LightningDataModule]
    ) -> None:
        if name == "naipchesapeake":
            pytest.importorskip("zipfile_deflate64")

        if name == "landcoverai":
            sha256 = "ecec8e871faf1bbd8ca525ca95ddc1c1f5213f40afb94599884bd85f990ebd6b"
            monkeypatch.setattr(LandCoverAI, "sha256", sha256)

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
        trainer = Trainer(
            logger=False, fast_dev_run=True, log_every_n_steps=1, max_epochs=1
        )
        trainer.fit(model=model, datamodule=datamodule)

    @pytest.fixture
    def model_kwargs(self) -> Dict[Any, Any]:
        return {
            "model": "unet",
            "backbone": "resnet18",
            "weights": None,
            "in_channels": 3,
            "num_classes": 6,
            "loss": "ce",
            "ignore_index": 0,
        }

    def test_invalid_model(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["model"] = "invalid_model"
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_invalid_loss(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["loss"] = "invalid_loss"
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_invalid_ignoreindex(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["ignore_index"] = "0"
        match = "ignore_index must be an int or None"
        with pytest.raises(ValueError, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_ignoreindex_with_jaccard(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["loss"] = "jaccard"
        model_kwargs["ignore_index"] = 0
        match = "ignore_index has no effect on training when loss='jaccard'"
        with pytest.warns(UserWarning, match=match):
            SemanticSegmentationTask(**model_kwargs)

    def test_missing_attributes(
        self, model_kwargs: Dict[Any, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delattr(LandCoverAI, "plot")
        datamodule = LandCoverAIDataModule(
            root="tests/data/landcoverai", batch_size=1, num_workers=0
        )
        model = SemanticSegmentationTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.validate(model=model, datamodule=datamodule)
