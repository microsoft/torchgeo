# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, cast

import pytest
from omegaconf import OmegaConf

from torchgeo.datamodules import LandCoverAIDataModule


class TestLandCoverAIDataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> LandCoverAIDataModule:
        conf = OmegaConf.load(os.path.join("conf", "task_defaults", "landcoverai.yaml"))
        kwargs = OmegaConf.to_object(conf.experiment.datamodule)
        kwargs = cast(Dict[str, Any], kwargs)

        datamodule = LandCoverAIDataModule(**kwargs)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule

    def test_undefined_attribute(self, datamodule: LandCoverAIDataModule) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
