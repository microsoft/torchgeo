# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Any

import pytest
import timm
import torch
import torchvision
from pytest import MonkeyPatch
from torch.nn import Module
from torchvision.models._api import WeightsEnum

from torchgeo.datasets import SSL4EOS12, SeasonalContrastS2
from torchgeo.main import main
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import SimCLRTask

from .test_classification import ClassificationTestModel


def create_model(*args: Any, **kwargs: Any) -> Module:
    return ClassificationTestModel(**kwargs)


def load(url: str, *args: Any, **kwargs: Any) -> dict[str, Any]:
    state_dict: dict[str, Any] = torch.load(url)
    return state_dict


class TestSimCLRTask:
    @pytest.mark.parametrize(
        "name",
        [
            "chesapeake_cvpr_prior_simclr",
            "seco_simclr_1",
            "seco_simclr_2",
            "ssl4eo_l_simclr_1",
            "ssl4eo_l_simclr_2",
            "ssl4eo_s12_simclr_1",
            "ssl4eo_s12_simclr_2",
        ],
    )
    def test_trainer(
        self, monkeypatch: MonkeyPatch, name: str, fast_dev_run: bool
    ) -> None:
        config = os.path.join("tests", "conf", name + ".yaml")

        if name.startswith("seco"):
            monkeypatch.setattr(SeasonalContrastS2, "__len__", lambda self: 2)

        if name.startswith("ssl4eo_s12"):
            monkeypatch.setattr(SSL4EOS12, "__len__", lambda self: 2)

        monkeypatch.setattr(timm, "create_model", create_model)

        args = [
            "--config",
            config,
            "--trainer.accelerator",
            "cpu",
            "--trainer.fast_dev_run",
            str(fast_dev_run),
            "--trainer.max_epochs",
            "1",
            "--trainer.log_every_n_steps",
            "1",
        ]

        main(["fit"] + args)

    def test_version_warnings(self) -> None:
        with pytest.warns(UserWarning, match="SimCLR v1 only uses 2 layers"):
            SimCLRTask(version=1, layers=3, memory_bank_size=0)
        with pytest.warns(UserWarning, match="SimCLR v1 does not use a memory bank"):
            SimCLRTask(version=1, layers=2, memory_bank_size=10)
        with pytest.warns(UserWarning, match=r"SimCLR v2 uses 3\+ layers"):
            SimCLRTask(version=2, layers=2, memory_bank_size=10)
        with pytest.warns(UserWarning, match="SimCLR v2 uses a memory bank"):
            SimCLRTask(version=2, layers=3, memory_bank_size=0)

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
        match = "num classes .* != num classes in pretrained model"
        with pytest.warns(UserWarning, match=match):
            SimCLRTask(model="resnet18", weights=checkpoint)

    def test_weight_enum(self, mocked_weights: WeightsEnum) -> None:
        match = "num classes .* != num classes in pretrained model"
        with pytest.warns(UserWarning, match=match):
            SimCLRTask(
                model=mocked_weights.meta["model"],
                weights=mocked_weights,
                in_channels=mocked_weights.meta["in_chans"],
            )

    def test_weight_str(self, mocked_weights: WeightsEnum) -> None:
        match = "num classes .* != num classes in pretrained model"
        with pytest.warns(UserWarning, match=match):
            SimCLRTask(
                model=mocked_weights.meta["model"],
                weights=str(mocked_weights),
                in_channels=mocked_weights.meta["in_chans"],
            )

    @pytest.mark.slow
    def test_weight_enum_download(self, weights: WeightsEnum) -> None:
        SimCLRTask(
            model=weights.meta["model"],
            weights=weights,
            in_channels=weights.meta["in_chans"],
        )

    @pytest.mark.slow
    def test_weight_str_download(self, weights: WeightsEnum) -> None:
        SimCLRTask(
            model=weights.meta["model"],
            weights=str(weights),
            in_channels=weights.meta["in_chans"],
        )
