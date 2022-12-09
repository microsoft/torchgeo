# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Type, cast

import pytest
import torch.nn as nn
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer
from torchvision.models import resnet18

from torchgeo.datamodules import ChesapeakeCVPRDataModule
from torchgeo.trainers import BYOLTask
from torchgeo.trainers.byol import BYOL, SimCLRAugmentation

from .test_utils import SegmentationTestModel


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
        conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

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
        trainer.test(model=model, datamodule=datamodule)
        trainer.predict(model=model, dataloaders=datamodule.val_dataloader())

    @pytest.fixture
    def model_kwargs(self) -> Dict[Any, Any]:
        return {"backbone": "resnet18", "weights": "random", "in_channels": 3}

    def test_invalid_pretrained(
        self, model_kwargs: Dict[Any, Any], checkpoint: str
    ) -> None:
        model_kwargs["weights"] = checkpoint
        model_kwargs["backbone"] = "resnet50"
        match = "Trying to load resnet18 weights into a resnet50"
        with pytest.raises(ValueError, match=match):
            BYOLTask(**model_kwargs)

    def test_pretrained(self, model_kwargs: Dict[Any, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        BYOLTask(**model_kwargs)

    def test_invalid_backbone(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["backbone"] = "invalid_backbone"
        match = "Model type 'invalid_backbone' is not a valid timm model."
        with pytest.raises(ValueError, match=match):
            BYOLTask(**model_kwargs)

    def test_invalid_weights(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["weights"] = "invalid_weights"
        match = "Weight type 'invalid_weights' is not valid."
        with pytest.raises(ValueError, match=match):
            BYOLTask(**model_kwargs)
