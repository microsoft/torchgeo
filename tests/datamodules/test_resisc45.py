# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, cast

import pytest
from omegaconf import OmegaConf

from torchgeo.datamodules import RESISC45DataModule


class TestRESISC45DataModule:
    @pytest.fixture(scope="class")
    def datamodule(self) -> RESISC45DataModule:
        conf = OmegaConf.load(os.path.join("conf", "task_defaults", "resisc45.yaml"))
        kwargs = OmegaConf.to_object(conf.experiment.datamodule)
        kwargs = cast(Dict[str, Any], kwargs)

        datamodule = RESISC45DataModule(**kwargs)
        datamodule.prepare_data()
        datamodule.setup()
        return datamodule

    def test_undefined_attribute(self, datamodule: RESISC45DataModule) -> None:
        batch = next(iter(datamodule.train_dataloader()))
        batch = datamodule.on_after_batch_transfer(batch, 0)
