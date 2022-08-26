# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Type, cast

import pytest
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer

from torchgeo.datamodules import COWCCountingDataModule, CycloneDataModule
from torchgeo.trainers import RegressionTask

from .test_utils import RegressionTestModel


class TestRegressionTask:
    @pytest.mark.parametrize(
        "name,classname",
        [("cowc_counting", COWCCountingDataModule), ("cyclone", CycloneDataModule)],
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
        model = RegressionTask(**model_kwargs)

        model.model = RegressionTestModel()

        # Instantiate trainer
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

    def test_no_logger(self) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", "cyclone.yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = CycloneDataModule(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = RegressionTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(
            logger=False, fast_dev_run=True, log_every_n_steps=1, max_epochs=1
        )
        trainer.fit(model=model, datamodule=datamodule)

    def test_invalid_model(self) -> None:
        match = "module 'torchvision.models' has no attribute 'invalid_model'"
        with pytest.raises(AttributeError, match=match):
            RegressionTask(model="invalid_model", pretrained=False)
