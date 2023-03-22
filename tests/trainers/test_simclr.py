# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Type, cast

import pytest
from _pytest.monkeypatch import MonkeyPatch
from lightning.pytorch import LightningDataModule, Trainer
from omegaconf import OmegaConf

from torchgeo.datamodules import ChesapeakeCVPRDataModule, SeasonalContrastS2DataModule
from torchgeo.datasets import SeasonalContrastS2
from torchgeo.trainers import SimCLRTask

from .test_classification import ClassificationTestModel


class TestSimCLRTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("chesapeake_cvpr_prior", ChesapeakeCVPRDataModule),
            ("seco_1", SeasonalContrastS2DataModule),
            ("seco_2", SeasonalContrastS2DataModule),
        ],
    )
    def test_trainer(
        self,
        monkeypatch: MonkeyPatch,
        name: str,
        classname: Type[LightningDataModule],
        fast_dev_run: bool,
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[str, Dict[str, Any]], conf_dict)

        if name.startswith("seco"):
            monkeypatch.setattr(SeasonalContrastS2, "__len__", lambda self: 2)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = SimCLRTask(**model_kwargs)
        model.model = ClassificationTestModel()

        # Instantiate trainer
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.fit(model=model, datamodule=datamodule)
