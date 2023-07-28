# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any, cast

import pytest
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
import torchvision
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import OmegaConf
from pytest import MonkeyPatch
from torch.nn.modules import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datamodules import MisconfigurationException, TropicalCycloneDataModule
from torchgeo.datasets import TropicalCyclone
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import PixelwiseRegressionTask, RegressionTask

from .test_classification import ClassificationTestModel


class PixelwiseRegressionTestModel(Module):
    def __init__(self, in_channels: int = 3, classes: int = 1, **kwargs: Any) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=classes, kernel_size=1, padding=0
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.conv1(x))


class RegressionTestModel(ClassificationTestModel):
    def __init__(self, in_chans: int = 3, num_classes: int = 1, **kwargs: Any) -> None:
        super().__init__(in_chans=in_chans, num_classes=num_classes)


class PredictRegressionDataModule(TropicalCycloneDataModule):
    def setup(self, stage: str) -> None:
        self.predict_dataset = TropicalCyclone(split="test", **self.kwargs)


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


def plot(*args: Any, **kwargs: Any) -> None:
    raise ValueError


def create_model(**kwargs: Any) -> Module:
    return PixelwiseRegressionTestModel(**kwargs)


class TestRegressionTask:
    @pytest.mark.parametrize(
        "name", ["cowc_counting", "cyclone", "sustainbench_crop_yield", "skippd"]
    )
    def test_trainer(self, name: str, fast_dev_run: bool) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))

        # Instantiate datamodule
        datamodule = instantiate(conf.datamodule)

        # Instantiate model
        model = instantiate(conf.module)

        model.model = RegressionTestModel(
            in_chans=conf.module.in_channels, num_classes=conf.module.num_outputs
        )

        # Instantiate trainer
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )

        trainer.fit(model=model, datamodule=datamodule)
        try:
            trainer.test(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass
        try:
            trainer.predict(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass

    @pytest.fixture
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "model": "resnet18",
            "weights": None,
            "num_outputs": 1,
            "in_channels": 3,
            "loss": "mse",
        }

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ResNet18_Weights.SENTINEL2_ALL_MOCO

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model(
            weights.meta["model"], in_chans=weights.meta["in_chans"]
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, "url", str(path))
        except AttributeError:
            monkeypatch.setattr(weights, "url", str(path))
        monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", load)
        return weights

    def test_weight_file(self, model_kwargs: dict[str, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        with pytest.warns(UserWarning):
            RegressionTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["model"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = mocked_weights
        with pytest.warns(UserWarning):
            RegressionTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["model"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = str(mocked_weights)
        with pytest.warns(UserWarning):
            RegressionTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_enum_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["model"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = weights
        RegressionTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_str_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["model"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = str(weights)
        RegressionTask(**model_kwargs)

    def test_no_rgb(
        self, monkeypatch: MonkeyPatch, model_kwargs: dict[Any, Any], fast_dev_run: bool
    ) -> None:
        monkeypatch.setattr(TropicalCycloneDataModule, "plot", plot)
        datamodule = TropicalCycloneDataModule(
            root="tests/data/cyclone", batch_size=1, num_workers=0
        )
        model = RegressionTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, model_kwargs: dict[Any, Any], fast_dev_run: bool) -> None:
        datamodule = PredictRegressionDataModule(
            root="tests/data/cyclone", batch_size=1, num_workers=0
        )
        model = RegressionTask(**model_kwargs)
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    def test_invalid_loss(self, model_kwargs: dict[str, Any]) -> None:
        model_kwargs["loss"] = "invalid_loss"
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            RegressionTask(**model_kwargs)

    @pytest.mark.parametrize(
        "model_name", ["resnet18", "efficientnetv2_s", "vit_base_patch16_384"]
    )
    def test_freeze_backbone(
        self, model_name: str, model_kwargs: dict[Any, Any]
    ) -> None:
        model_kwargs["freeze_backbone"] = True
        model_kwargs["model"] = model_name
        model = RegressionTask(**model_kwargs)
        assert not all([param.requires_grad for param in model.model.parameters()])
        assert all(
            [param.requires_grad for param in model.model.get_classifier().parameters()]
        )


class TestPixelwiseRegressionTask:
    @pytest.mark.parametrize(
        "name,batch_size,loss,model_type",
        [
            ("inria", 1, "mse", "unet"),
            ("inria", 2, "mae", "deeplabv3+"),
            ("inria", 1, "mse", "fcn"),
        ],
    )
    def test_trainer(
        self,
        monkeypatch: MonkeyPatch,
        name: str,
        batch_size: int,
        loss: str,
        model_type: str,
        fast_dev_run: bool,
        model_kwargs: dict[str, Any],
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))

        # Instantiate datamodule
        conf.datamodule.batch_size = batch_size
        datamodule = instantiate(conf.datamodule)

        # Instantiate model
        monkeypatch.setattr(smp, "Unet", create_model)
        monkeypatch.setattr(smp, "DeepLabV3Plus", create_model)
        model_kwargs["model"] = model_type
        model_kwargs["loss"] = loss

        if model_type == "fcn":
            model_kwargs["num_filters"] = 2

        model = PixelwiseRegressionTask(**model_kwargs)
        model.model = PixelwiseRegressionTestModel(
            in_channels=model_kwargs["in_channels"]
        )

        # Instantiate trainer
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )

        trainer.fit(model=model, datamodule=datamodule)
        try:
            trainer.test(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass
        try:
            trainer.predict(model=model, datamodule=datamodule)
        except MisconfigurationException:
            pass

    def test_invalid_model(self, model_kwargs: dict[str, Any]) -> None:
        model_kwargs["model"] = "invalid_model"
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            PixelwiseRegressionTask(**model_kwargs)

    @pytest.fixture
    def model_kwargs(self) -> dict[str, Any]:
        return {
            "model": "unet",
            "backbone": "resnet18",
            "weights": None,
            "num_outputs": 1,
            "in_channels": 3,
            "loss": "mse",
            "learning_rate": 1e-3,
            "learning_rate_schedule_patience": 6,
        }

    @pytest.fixture
    def weights(self) -> WeightsEnum:
        return ResNet18_Weights.SENTINEL2_ALL_MOCO

    @pytest.fixture
    def mocked_weights(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, weights: WeightsEnum
    ) -> WeightsEnum:
        path = tmp_path / f"{weights}.pth"
        model = timm.create_model(
            weights.meta["model"], in_chans=weights.meta["in_chans"]
        )
        torch.save(model.state_dict(), path)
        try:
            monkeypatch.setattr(weights.value, "url", str(path))
        except AttributeError:
            monkeypatch.setattr(weights, "url", str(path))
        monkeypatch.setattr(torchvision.models._api, "load_state_dict_from_url", load)
        return weights

    def test_weight_file(self, model_kwargs: dict[str, Any], checkpoint: str) -> None:
        model_kwargs["weights"] = checkpoint
        PixelwiseRegressionTask(**model_kwargs)

    def test_weight_enum(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = mocked_weights
        PixelwiseRegressionTask(**model_kwargs)

    def test_weight_str(
        self, model_kwargs: dict[str, Any], mocked_weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = mocked_weights.meta["model"]
        model_kwargs["in_channels"] = mocked_weights.meta["in_chans"]
        model_kwargs["weights"] = str(mocked_weights)
        PixelwiseRegressionTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_enum_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = weights
        PixelwiseRegressionTask(**model_kwargs)

    @pytest.mark.slow
    def test_weight_str_download(
        self, model_kwargs: dict[str, Any], weights: WeightsEnum
    ) -> None:
        model_kwargs["backbone"] = weights.meta["model"]
        model_kwargs["in_channels"] = weights.meta["in_chans"]
        model_kwargs["weights"] = str(weights)
        PixelwiseRegressionTask(**model_kwargs)

    @pytest.mark.parametrize(
        "backbone", ["resnet18", "mobilenet_v2", "efficientnet-b0"]
    )
    @pytest.mark.parametrize("model_name", ["unet", "deeplabv3+"])
    def test_freeze_backbone(
        self, backbone: str, model_name: str, model_kwargs: dict[Any, Any]
    ) -> None:
        model_kwargs["freeze_backbone"] = True
        model_kwargs["model"] = model_name
        model_kwargs["backbone"] = backbone
        model = PixelwiseRegressionTask(**model_kwargs)
        assert all(
            [param.requires_grad is False for param in model.model.encoder.parameters()]
        )
        assert all([param.requires_grad for param in model.model.decoder.parameters()])
        assert all(
            [
                param.requires_grad
                for param in model.model.segmentation_head.parameters()
            ]
        )

    @pytest.mark.parametrize("model_name", ["unet", "deeplabv3+"])
    def test_freeze_decoder(
        self, model_name: str, model_kwargs: dict[Any, Any]
    ) -> None:
        model_kwargs["freeze_decoder"] = True
        model_kwargs["model"] = model_name
        model = PixelwiseRegressionTask(**model_kwargs)
        assert all(
            [param.requires_grad is False for param in model.model.decoder.parameters()]
        )
        assert all([param.requires_grad for param in model.model.encoder.parameters()])
        assert all(
            [
                param.requires_grad
                for param in model.model.segmentation_head.parameters()
            ]
        )
