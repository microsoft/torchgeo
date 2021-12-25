# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, cast

import pytest
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, Trainer

from torchgeo.datamodules import COWCCountingDataModule, CycloneDataModule
from torchgeo.trainers import RegressionTask


class TestRegressionTask:
    @pytest.mark.parametrize(
        "name,classname,datamodule_kwargs",
        [
            ("cowc_counting", COWCCountingDataModule, {}),
            ("cyclone", CycloneDataModule, {}),
        ],
    )
    def test_trainer(
        self,
        name: str,
        classname: LightningDataModule,
        datamodule_kwargs: Dict[Any, Any],
    ) -> None:
        # Instantiate datamodule
        root = os.path.join("tests", "data", name)
        datamodule = classname(
            root, seed=0, batch_size=1, num_workers=0, **datamodule_kwargs
        )

        # Instantiate model
        model_conf = OmegaConf.load(
            os.path.join("conf", "task_defaults", name + ".yaml")
        )
        model_kwargs = OmegaConf.to_object(model_conf.experiment.module)
        model_kwargs = cast(Dict[str, Any], model_kwargs)
        model = RegressionTask(**model_kwargs)

        # Instantiate trainer
        trainer = Trainer(fast_dev_run=True, log_every_n_steps=1)
        trainer.fit(model=model, datamodule=datamodule)
        trainer.test(model=model, datamodule=datamodule)

    def test_invalid_model(self) -> None:
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            RegressionTask(model="invalid_model")
