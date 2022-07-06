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

from .test_utils import ClassificationTestModel


class TestBYOL:
    def test_custom_augment_fn(self) -> None:
        encoder = resnet18()
        layer = encoder.conv1
        new_layer = nn.Conv2d(
            in_channels=4,
            out_channels=layer.out_channels,
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            bias=layer.bias,
        ).requires_grad_()
        encoder.conv1 = new_layer
        augment_fn = SimCLRAugmentation((2, 2))
        BYOL(encoder, augment_fn=augment_fn)


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

        model.encoder = ClassificationTestModel(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

    def test_invalid_encoder(self) -> None:
        kwargs = {
            "in_channels": 1,
            "imagenet_pretraining": False,
            "encoder_name": "invalid_encoder",
        }
        error_message = "module 'torchvision.models' has no attribute 'invalid_encoder'"
        with pytest.raises(AttributeError, match=error_message):
            BYOLTask(**kwargs)
