# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Type, cast

import pytest
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer

from torchgeo.datamodules import COWCCountingDataModule, TropicalCycloneDataModule
from torchgeo.trainers import RegressionTask

from .test_utils import RegressionTestModel


class TestRegressionTask:
    @pytest.mark.parametrize(
        "name,classname",
        [
            ("cowc_counting", COWCCountingDataModule),
            ("cyclone", TropicalCycloneDataModule),
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
        model = RegressionTask(**model_kwargs)

        model.model = RegressionTestModel()

        # Instantiate trainer
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
        trainer.predict(model=model, dataloaders=datamodule.val_dataloader())

    def test_no_logger(self) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", "cyclone.yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = TropicalCycloneDataModule(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = RegressionTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(
            logger=False, fast_dev_run=True, log_every_n_steps=1, max_epochs=1
        )
        trainer.fit(model=model, datamodule=datamodule)

    @pytest.fixture
    def model_kwargs(self) -> Dict[Any, Any]:
        return {
            "model": "resnet18",
            "weights": "random",
            "num_outputs": 1,
            "in_channels": 3,
        }

    def test_invalid_pretrained(
        self, model_kwargs: Dict[Any, Any], checkpoint: str
    ) -> None:
        model_kwargs["weights"] = checkpoint
        model_kwargs["model"] = "resnet50"
        match = "Trying to load resnet18 weights into a resnet50"
        with pytest.raises(ValueError, match=match):
            RegressionTask(**model_kwargs)

    def test_pretrained(self, model_kwargs: Dict[Any, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        with pytest.warns(UserWarning):
            RegressionTask(**model_kwargs)

    def test_invalid_model(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["model"] = "invalid_model"
        match = "Model type 'invalid_model' is not a valid timm model."
        with pytest.raises(ValueError, match=match):
            RegressionTask(**model_kwargs)

    def test_invalid_weights(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["weights"] = "invalid_weights"
        match = "Weight type 'invalid_weights' is not valid."
        with pytest.raises(ValueError, match=match):
            RegressionTask(**model_kwargs)
