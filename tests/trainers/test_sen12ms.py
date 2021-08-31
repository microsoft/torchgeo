# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Dict, cast

import pytest
import segmentation_models_pytorch as smp
import torch.nn as nn
from omegaconf import OmegaConf

from torchgeo.trainers import SEN12MSDataModule, SEN12MSSegmentationTask


class TestSEN12MSTrainer:
    @pytest.fixture
    def default_config(self) -> Dict[str, Any]:
        task_conf = OmegaConf.load("conf/task_defaults/sen12ms.yaml")
        task_args = OmegaConf.to_object(task_conf.task)
        task_args = cast(Dict[str, Any], task_args)
        return task_args

    def test_unet_ce(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "unet"
        default_config["loss"] = "ce"
        default_config["optimizer"] = "adamw"
        task = SEN12MSSegmentationTask(**default_config)
        assert isinstance(task.model, smp.Unet)
        assert isinstance(task.loss, nn.CrossEntropyLoss)  # type: ignore[attr-defined]

    def test_unet_jaccard(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "unet"
        default_config["loss"] = "jaccard"
        task = SEN12MSSegmentationTask(**default_config)
        assert isinstance(task.model, smp.Unet)
        assert isinstance(task.loss, smp.losses.JaccardLoss)

    def test_invalid_model(self, default_config: Dict[str, Any]) -> None:
        default_config["segmentation_model"] = "invalid_model"
        error_message = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=error_message):
            SEN12MSSegmentationTask(**default_config)

    def test_invalid_loss(self, default_config: Dict[str, Any]) -> None:
        default_config["loss"] = "invalid_loss"
        error_message = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=error_message):
            SEN12MSSegmentationTask(**default_config)


@pytest.mark.parametrize("band_set", ["all", "s1", "s2-all", "s2-reduced"])
def test_band_set(band_set: str) -> None:
    dm = SEN12MSDataModule(os.path.join("tests", "data", "sen12ms"), 0, band_set)
    dm.prepare_data()
    dm.all_train_dataset[0]
