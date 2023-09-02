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

    def test_weight_file(self, checkpoint: str) -> None:
        with pytest.warns(UserWarning):
            RegressionTask(model="resnet18", weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        with pytest.warns(UserWarning):
            RegressionTask(
                model=mocked_weights.meta["model"],
                weights=mocked_weights,
                in_channels=mocked_weights.meta["in_chans"],
            )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        with pytest.warns(UserWarning):
            RegressionTask(
                model=mocked_weights.meta["model"],
                weights=str(mocked_weights),
                in_channels=mocked_weights.meta["in_chans"],
            )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        RegressionTask(
            model=weights.meta["model"],
            weights=weights,
            in_channels=weights.meta["in_chans"],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        RegressionTask(
            model=weights.meta["model"],
            weights=str(weights),
            in_channels=weights.meta["in_chans"],
        )

    def test_no_rgb(self, monkeypatch: MonkeyPatch, fast_dev_run: bool) -> None:
        monkeypatch.setattr(TropicalCycloneDataModule, "plot", plot)
        datamodule = TropicalCycloneDataModule(
            root="tests/data/cyclone", batch_size=1, num_workers=0
        )
        model = RegressionTask(model="resnet18")
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.validate(model=model, datamodule=datamodule)

    def test_predict(self, fast_dev_run: bool) -> None:
        datamodule = PredictRegressionDataModule(
            root="tests/data/cyclone", batch_size=1, num_workers=0
        )
        model = RegressionTask(model="resnet18")
        trainer = Trainer(
            accelerator="cpu",
            fast_dev_run=fast_dev_run,
            log_every_n_steps=1,
            max_epochs=1,
        )
        trainer.predict(model=model, datamodule=datamodule)

    def test_invalid_loss(self) -> None:
        match = "Loss type 'invalid_loss' is not valid."
        with pytest.raises(ValueError, match=match):
            RegressionTask(model="resnet18", loss="invalid_loss")

    @pytest.mark.parametrize(
        "model_name", ["resnet18", "efficientnetv2_s", "vit_base_patch16_384"]
    )
    def test_freeze_backbone(self, model_name: str) -> None:
        model = RegressionTask(model=model_name, freeze_backbone=True)
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
    ) -> None:
        conf = OmegaConf.load(os.path.join("tests", "conf", name + ".yaml"))

        # Instantiate datamodule
        conf.datamodule.batch_size = batch_size
        datamodule = instantiate(conf.datamodule)

        # Instantiate model
        monkeypatch.setattr(smp, "Unet", create_model)
        monkeypatch.setattr(smp, "DeepLabV3Plus", create_model)

        model = PixelwiseRegressionTask(
            model=model_type, backbone="resnet18", loss=loss
        )
        model.model = PixelwiseRegressionTestModel()

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

    def test_invalid_model(self) -> None:
        match = "Model type 'invalid_model' is not valid."
        with pytest.raises(ValueError, match=match):
            PixelwiseRegressionTask(model="invalid_model")

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

    def test_weight_file(self, checkpoint: str) -> None:
        PixelwiseRegressionTask(model="unet", backbone="resnet18", weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        PixelwiseRegressionTask(
            model="unet",
            backbone=mocked_weights.meta["model"],
            weights=mocked_weights,
            in_channels=mocked_weights.meta["in_chans"],
        )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        PixelwiseRegressionTask(
            model="unet",
            backbone=mocked_weights.meta["model"],
            weights=str(mocked_weights),
            in_channels=mocked_weights.meta["in_chans"],
        )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        PixelwiseRegressionTask(
            model="unet",
            backbone=weights.meta["model"],
            weights=weights,
            in_channels=weights.meta["in_chans"],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        PixelwiseRegressionTask(
            model="unet",
            backbone=weights.meta["model"],
            weights=str(weights),
            in_channels=weights.meta["in_chans"],
        )

    @pytest.mark.parametrize("model_name", ["unet", "deeplabv3+"])
    @pytest.mark.parametrize(
        "backbone", ["resnet18", "mobilenet_v2", "efficientnet-b0"]
    )
    def test_freeze_backbone(self, model_name: str, backbone: str) -> None:
        model = PixelwiseRegressionTask(
            model=model_name, backbone=backbone, freeze_backbone=True
        )
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
    def test_freeze_decoder(self, model_name: str) -> None:
        model = PixelwiseRegressionTask(
            model=model_name, backbone="resnet18", freeze_decoder=True
        )
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
