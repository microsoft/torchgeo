# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, cast

import pytest
import timm
from _pytest.monkeypatch import MonkeyPatch
from lightning.pytorch import LightningDataModule, Trainer
from omegaconf import OmegaConf
from torch.nn import Module

from torchgeo.datamodules import (
    ChesapeakeCVPRDataModule,
    SeasonalContrastS2DataModule,
    SSL4EOS12DataModule,
)
from torchgeo.datasets import SSL4EOS12, SeasonalContrastS2
from torchgeo.trainers import SimCLRTask

from .test_classification import ClassificationTestModel


def create_model(*args: Any, **kwargs: Any) -> Module:
    return ClassificationTestModel(**kwargs)


class TestSimCLRTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("chesapeake_cvpr_prior_simclr_1", ChesapeakeCVPRDataModule),
            ("chesapeake_cvpr_prior_simclr_2", ChesapeakeCVPRDataModule),
            ("seco_simclr_1", SeasonalContrastS2DataModule),
            ("seco_simclr_2", SeasonalContrastS2DataModule),
            ("ssl4eo_s12_simclr_1", SSL4EOS12DataModule),
            ("ssl4eo_s12_simclr_2", SSL4EOS12DataModule),
        ],
    )
    def test_trainer(
        self,
        monkeypatch: MonkeyPatch,
        name: str,
        classname: type[LightningDataModule],
        fast_dev_run: bool,
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(dict[str, dict[str, Any]], conf_dict)

        if name.startswith("seco"):
            monkeypatch.setattr(SeasonalContrastS2, "__len__", lambda self: 2)

        if name.startswith("ssl4eo_s12"):
            monkeypatch.setattr(SSL4EOS12, "__len__", lambda self: 2)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        monkeypatch.setattr(timm, "create_model", create_model)
        model_kwargs = conf_dict["module"]
        model = SimCLRTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model=model, datamodule=datamodule)
