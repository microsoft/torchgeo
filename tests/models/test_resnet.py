# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, Dict, cast

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
from torchgeo.models import ResNet18_Weights, ResNet50_Weights
from torchgeo.models.weights import lookup_pretrained_weights
from torchgeo.trainers import ClassificationTask, RegressionTask


def load_state_dict_from_url(
    root: str, filename: str, url: str, map_location: torch.device
) -> Any:
    """Mockup of ``torchgeo.models.resnet.load_state_dict_from_url."""
    return torch.load(url)


def adjust_moco_state_dict(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Adjust the moco weight names."""
    new_state_dict = {"module.encoder_q." + key: val for key, val in state_dict.items()}
    return new_state_dict


# RESNET18 Weights
@pytest.fixture
def resnet18_sentinel2_rgb_moco(tmp_path: Path) -> str:
    num_input_channels = 3
    weight_key = "SENTINEL2_RGB_MOCO"
    model = timm.create_model("resnet18", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet18_{weight_key.lower()}.pth")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet18_sentinel2_all_moco(tmp_path: Path) -> str:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_MOCO"
    model = timm.create_model("resnet18", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet18_{weight_key.lower()}.pth")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.mark.parametrize(
    "generate_model,weight",
    [
        ("resnet18_sentinel2_rgb_moco", ResNet18_Weights.SENTINEL2_RGB_MOCO),
        ("resnet18_sentinel2_all_moco", ResNet18_Weights.SENTINEL2_ALL_MOCO),
    ],
)
def test_resnet18_pretrained_weights(
    monkeypatch: MonkeyPatch, request: SubRequest, generate_model, weight
) -> None:

    ckpt_path, num_input_channels = request.getfixturevalue(generate_model)

    monkeypatch.setattr(weight, "url", ckpt_path)
    monkeypatch.setattr(
        torchgeo.models.resnet, "load_state_dict_from_url", load_state_dict_from_url
    )

    task = ClassificationTask(
        model="resnet18",
        loss="ce",
        in_channels=num_input_channels,
        weights=weight.get_state_dict(),
        num_classes=1000,  # imagenet default weights timm
    )
    x = torch.zeros(1, num_input_channels, 64, 64)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)


@pytest.mark.parametrize(
    "config_filename, datamodule_name, generate_model",
    [("resnet18_weight.yaml", EuroSATDataModule, "resnet18_sentinel2_all_moco")],
)
def test_pretrained_resnet18_from_config(
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
@pytest.mark.parametrize("weight_name", [(w.name) for w in ResNet18_Weights])
@pytest.mark.parametrize(
    "task, task_args",
    [
        (ClassificationTask, {"model": "resnet18", "loss": "ce", "num_classes": 1000}),
        (RegressionTask, {"model": "resnet18", "loss": "mse", "num_outputs": 1000}),
    ],
)
def test_resnet18_weights_download(
    weight_name: str, task: pl.LightningModule, task_args: Dict[str, Any]
) -> None:
    weight = ResNet18_Weights[weight_name]
    num_input_channels = weight.meta["num_input_channels"]

    task = task(
        in_channels=num_input_channels, weights=weight.get_state_dict(), **task_args
    )
    x = torch.zeros(2, num_input_channels, 64, 64)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)


# RESNET 50 Weights
@pytest.fixture
def resnet50_sentinel2_rgb_moco(tmp_path: Path) -> str:
    num_input_channels = 3
    weight_key = "SENTINEL2_RGB_MOCO"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pth")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet50_sentinel2_all_moco(tmp_path: Path) -> str:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_MOCO"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pth")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet50_sentinel1_grd_moco(tmp_path: Path) -> str:
    num_input_channels = 2
    weight_key = "SENTINEL1_GRD_MOCO"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pth")
    ckpt_dict = {"state_dict": adjust_moco_state_dict(model.state_dict())}
    torch.save(ckpt_dict, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.fixture
def resnet50_sentinel2_all_dino(tmp_path: Path) -> str:
    num_input_channels = 13
    weight_key = "SENTINEL2_ALL_DINO"
    model = timm.create_model("resnet50", in_chans=num_input_channels)
    ckpt_path = os.path.join(tmp_path, f"resnet50_{weight_key.lower()}.pth")
    torch.save({"teacher": model.state_dict()}, ckpt_path)
    return ckpt_path, num_input_channels


@pytest.mark.parametrize(
    "generate_model,weight",
    [
        (
            "resnet50_googleearth_millionaid_rgb",
            ResNet50_Weights.GOOGLEEARTH_MILLIONAID_RGB,
        ),
        ("resnet50_sentinel2_rgb_moco", ResNet50_Weights.SENTINEL2_RGB_MOCO),
        ("resnet50_sentinel2_all_moco", ResNet50_Weights.SENTINEL2_ALL_MOCO),
        ("resnet50_sentinel1_grd_moco", ResNet50_Weights.SENTINEL1_GRD_MOCO),
        ("resnet50_sentinel2_all_dino", ResNet50_Weights.SENTINEL2_ALL_DINO),
    ],
)
@pytest.mark.parametrize(
    "task, task_args",
    [
        (ClassificationTask, {"model": "resnet50", "loss": "ce", "num_classes": 1000}),
        (RegressionTask, {"model": "resnet50", "loss": "mse", "num_outputs": 1000}),
    ],
)
def test_resnet50_pretrained_weights(
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
        torchgeo.models.resnet, "load_state_dict_from_url", load_state_dict_from_url
    )

    task = task(
        in_channels=num_input_channels, weights=weight.get_state_dict(), **task_args
    )
    x = torch.zeros(2, num_input_channels, 64, 64)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)


@pytest.mark.parametrize(
    "config_filename, datamodule_name, generate_model",
    [("resnet50_weight.yaml", EuroSATDataModule, "resnet50_sentinel2_all_moco")],
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
@pytest.mark.parametrize("weight_name", [(w.name) for w in ResNet50_Weights])
@pytest.mark.parametrize(
    "task, task_args",
    [
        (ClassificationTask, {"model": "resnet50", "loss": "ce", "num_classes": 1000}),
        (RegressionTask, {"model": "resnet50", "loss": "mse", "num_outputs": 1000}),
    ],
)
def test_resnet50_weights_download(
    weight_name: str, task: pl.LightningModule, task_args: Dict[str, Any]
) -> None:
    weight = ResNet50_Weights[weight_name]
    num_input_channels = weight.meta["num_input_channels"]

    task = task(
        in_channels=num_input_channels, weights=weight.get_state_dict(), **task_args
    )
    x = torch.zeros(2, num_input_channels, 64, 64)
    y = task.forward(x)
    assert isinstance(y, torch.Tensor)
