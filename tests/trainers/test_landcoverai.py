# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from typing import Any, Dict, cast

import pytest
import segmentation_models_pytorch as smp
import torch.nn as nn
from omegaconf import OmegaConf

import torchgeo.models
from torchgeo.trainers import LandcoverAISegmentationTask


class TestLandCoverAITrainer:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load("conf/task_defaults/landcoverai.yaml")
        task_args = OmegaConf.to_object(task_conf.experiment.module)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    def test_unet_ce(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "unet"
        default_config["loss"] = "ce"
        task = LandcoverAISegmentationTask(**default_config)
        assert isinstance(task.model, smp.Unet)
        assert isinstance(task.loss, nn.CrossEntropyLoss)  # type: ignore[attr-defined]

    def test_fcn_jaccard(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "fcn"
        default_config["loss"] = "jaccard"
        task = LandcoverAISegmentationTask(**default_config)
        assert isinstance(task.model, torchgeo.models.FCN)
        assert isinstance(task.loss, smp.losses.JaccardLoss)

    def test_invalid_model(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            LandcoverAISegmentationTask(**default_config)

    def test_invalid_loss(self, default_config: Dict[str, Any]) -> None:
        default_config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            LandcoverAISegmentationTask(**default_config)
