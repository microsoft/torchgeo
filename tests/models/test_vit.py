# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, Dict, Tuple, cast

import kornia.augmentation as K
import pytest
import pytorch_lightning as pl
import timm
import torch
from _pytest.fixtures import SubRequest
from _pytest.monkeypatch import MonkeyPatch
from omegaconf import OmegaConf
from torch import Tensor

import torchgeo.models.resnet
from torchgeo.datamodules import EuroSATDataModule
from torchgeo.models import VITSmall16_Weights
from torchgeo.models.weights import lookup_pretrained_weights
from torchgeo.trainers import ClassificationTask, RegressionTask
from torchgeo.transforms import AugmentationSequential


def load_state_dict_from_url(
    root: str, filename: str, url: str, map_location: torch.device
) -> Any:
    """Mockup of ``torchgeo.models.resnet.load_state_dict_from_url."""
    return torch.load(url)


def adjust_moco_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Adjust the moco weight names."""
    new_state_dict = {"module.encoder_q." + key: val for key, val in state_dict.items()}
    return new_state_dict


def custom_augmentation(self, sample: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Custom augmentation to patch datamodules during testing."""
    sample["image"] = sample["image"].float()
    transform = AugmentationSequential(
        K.Normalize(mean=0.0, std=255.0), K.Resize(224), data_keys=["image"]
    )
    out = transform(sample)
    out["image"] = out["image"].squeeze(0)
    return out


# VITSmall16 weights
@pytest.fixture
def vitsmall16_sentinel2_all_moco(tmp_path: Path) -> Tuple[str, int]:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_MOCO"
    model = timm.create_model("vit_small_patch16_224", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"vitsmall16_{weight_key.lower()}.pt")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def vitsmall16_sentinel2_all_dino(tmp_path: Path) -> Tuple[str, int]:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_DINO"
    model = timm.create_model("vit_small_patch16_224", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"vitsmall16_{weight_key.lower()}.pt")
    torch.save({"teacher": model.state_dict()}, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.mark.parametrize(
    "generate_model, weight",
    [
        ("vitsmall16_sentinel2_all_moco", VITSmall16_Weights.SENTINEL2_ALL_MOCO),
        ("vitsmall16_sentinel2_all_dino", VITSmall16_Weights.SENTINEL2_ALL_DINO),
    ],
)
@pytest.mark.parametrize(
    "task, task_args",
    [
        (
            ClassificationTask,
            {"model": "vit_small_patch16_224", "loss": "ce", "num_classes": 1000},
        ),
        (
            RegressionTask,
            {"model": "vit_small_patch16_224", "loss": "mse", "num_outputs": 1000},
        ),
    ],
)
def test_vitsmall16_pretrained_weights(
    monkeypatch: MonkeyPatch,
    request: SubRequest,
    generate_model,
    weight,
    task: pl.LightningModule,
    task_args: Dict[str, Any],
) -> None:

    ckpt_path, num_input_channels = request.getfixturevalue(generate_model)

    monkeypatch.setattr(weight, "url", ckpt_path)
    monkeypatch.setattr(
        torchgeo.models.vit, "load_state_dict_from_url", load_state_dict_from_url
    )

    task = task(
        in_channels=num_input_channels, weights=weight.get_state_dict(), **task_args
    )
    x = torch.zeros(2, num_input_channels, 224, 224)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)


@pytest.mark.parametrize(
    "config_filename, datamodule_name, generate_model",
    [
        (
            "vit_small_patch16_224_weight.yaml",
            EuroSATDataModule,
            "vitsmall16_sentinel2_all_moco",
        )
    ],
)
def test_pretrained_resnet50_from_config(
    monkeypatch: MonkeyPatch,
    request: SubRequest,
    config_filename: str,
    datamodule_name: pl.LightningDataModule,
    generate_model,
) -> None:
    conf = OmegaConf.load(os.path.join("tests", "conf", config_filename))
    conf_dict = OmegaConf.to_object(conf.experiment)
    conf_dict = cast(Dict[Any, Dict[Any, Any]], conf_dict)

    # Instantiate datamodule
    datamodule_kwargs = conf_dict["datamodule"]
    # input size to vit_small_patch16_224 needs to be (224, 224)
    monkeypatch.setattr(datamodule_name, "preprocess", custom_augmentation)
    datamodule = datamodule_name(**datamodule_kwargs)

    # Instantiate model
    ckpt_path, _ = request.getfixturevalue(generate_model)
    weight = lookup_pretrained_weights(
        conf_dict["module"]["model"], conf_dict["module"]["weights"]
    )
    monkeypatch.setattr(weight, "url", ckpt_path)
    monkeypatch.setattr(
        torchgeo.models.resnet, "load_state_dict_from_url", load_state_dict_from_url
    )
    model_kwargs = {
        key: val for key, val in conf_dict["module"].items() if key not in ["weights"]
    }
    model = ClassificationTask(weights=weight.get_state_dict(), **model_kwargs)

    # Instantiate trainer
    trainer = pl.Trainer(fast_dev_run=True, log_every_n_steps=1, max_epochs=1)
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule)
    trainer.predict(model=model, dataloaders=datamodule.val_dataloader())


@pytest.mark.slow
@pytest.mark.parametrize("weight_name", [(w.name) for w in VITSmall16_Weights])
@pytest.mark.parametrize(
    "task, task_args",
    [
        (
            ClassificationTask,
            {"model": "vit_small_patch16_224", "loss": "ce", "num_classes": 1000},
        ),
        (
            RegressionTask,
            {"model": "vit_small_patch16_224", "loss": "mse", "num_outputs": 1000},
        ),
    ],
)
def test_vit_small_patch16_224_weights_download(
    weight_name: str, task: pl.LightningModule, task_args: Dict[str, Any]
) -> None:
    weight = VITSmall16_Weights[weight_name]
    num_input_channels = weight.meta["num_input_channels"]

    task = task(
        in_channels=num_input_channels, weights=weight.get_state_dict(), **task_args
    )
    x = torch.zeros(1, num_input_channels, 224, 224)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)
