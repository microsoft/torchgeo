# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict, cast

import pytest
import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision
from omegaconf import OmegaConf

from torchgeo.trainers import So2SatClassificationTask


class TestSo2SatTrainer:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load("conf/task_defaults/so2sat.yaml")
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    def test_resnet_ce(self, default_config: Dict[str, Any]) -> None:
        default_config["classification_model"] = "resnet18"
        default_config["loss"] = "ce"
        task = So2SatClassificationTask(**default_config)
        assert isinstance(task.model, torchvision.models.ResNet)
        assert isinstance(task.loss, nn.CrossEntropyLoss)  # type: ignore[attr-defined]

    def test_resnet_jaccard(self, default_config: Dict[str, Any]) -> None:
        default_config["classification_model"] = "resnet18"
        default_config["loss"] = "jaccard"
        task = So2SatClassificationTask(**default_config)
        assert isinstance(task.model, torchvision.models.ResNet)
        assert isinstance(task.loss, smp.losses.JaccardLoss)

    def test_resnet_focal(self, default_config: Dict[str, Any]) -> None:
        default_config["classification_model"] = "resnet18"
        default_config["loss"] = "focal"
        task = So2SatClassificationTask(**default_config)
        assert isinstance(task.model, torchvision.models.ResNet)
        assert isinstance(task.loss, smp.losses.FocalLoss)

    def test_invalid_model(self, default_config: Dict[str, Any]) -> None:
        default_config["classification_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            So2SatClassificationTask(**default_config)

    def test_invalid_loss(self, default_config: Dict[str, Any]) -> None:
        default_config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            So2SatClassificationTask(**default_config)
