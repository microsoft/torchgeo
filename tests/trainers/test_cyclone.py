# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict, cast

import pytest
from omegaconf import OmegaConf
from torchvision import models

from torchgeo.trainers import CycloneSimpleRegressionTask


class TestCycloneTrainer:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load("conf/task_defaults/cyclone.yaml")
        task_args = OmegaConf.to_object(task_conf.task)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    def test_resnet18(self, default_config: Dict[str, Any]) -> None:
        default_config["model"] = "resnet18"
        task = CycloneSimpleRegressionTask(**default_config)
        assert isinstance(task.model, models.resnet.ResNet)

    def test_invalid_model(self, default_config: Dict[str, Any]) -> None:
        default_config["model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            CycloneSimpleRegressionTask(**default_config)
