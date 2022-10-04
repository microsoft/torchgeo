# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, Type, cast

import pytest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer

from torchgeo.datamodules import NASAMarineDebrisDataModule
from torchgeo.trainers import ObjectDetectionTask


class TestObjectDetectionTask:
    @pytest.mark.parametrize(
        "name,classname", [("nasa_marine_debris", NASAMarineDebrisDataModule)]
    )
    def test_trainer(self, name: str, classname: Type[LightningDataModule]) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", f"{name}.yaml"))
        conf_dict = OmegaConf.to_object(conf.experiment)
        conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

        # Instantiate datamodule
        datamodule_kwargs = conf_dict["datamodule"]
        datamodule = classname(**datamodule_kwargs)

        # Instantiate model
        model_kwargs = conf_dict["module"]
        model = ObjectDetectionTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)
        trainer.predict(model=model, dataloaders=datamodule.val_dataloader())

    @pytest.fixture
    def model_kwargs(self) -> Dict[Any, Any]:
        return {
            "detection_model": "faster-rcnn",
            "backbone": "resnet18",
            "num_classes": 2,
        }

    def test_invalid_model(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["detection_model"] = "invalid_model"
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetectionTask(**model_kwargs)

    def test_invalid_backbone(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["backbone"] = "invalid_backbone"
        match = "Backbone type 'invalid_backbone' is not valid."
        with pytest.raises(ValueError, match=match):
            ObjectDetectionTask(**model_kwargs)

    def test_non_pretrained_backbone(self, model_kwargs: Dict[Any, Any]) -> None:
        model_kwargs["pretrained"] = False
        ObjectDetectionTask(**model_kwargs)

    def test_missing_attributes(
        self, model_kwargs: Dict[Any, Any], monkeypatch: MonkeyPatch
    ) -> None:
        monkeypatch.delattr(NASAMarineDebrisDataModule, "plot")
        datamodule = NASAMarineDebrisDataModule(
            root="tests/data/nasa_marine_debris", batch_size=1, num_workers=0
        )
        model = ObjectDetectionTask(**model_kwargs)
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
        trainer.validate(model=model, datamodule=datamodule)
