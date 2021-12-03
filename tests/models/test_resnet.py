# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Generator

import pytest
import torch
from _pytest.monkeypatch import MonkeyPatch
from torch.nn.modules import Module

import torchgeo.models.resnet
from torchgeo.datasets.utils import extract_archive
from torchgeo.models import resnet50


@pytest.mark.parametrize(
    "model_class,sensor,bands,in_channels,num_classes",
    [(resnet50, "sentinel2", "all", 10, 17)],
)
def test_resnet(
    monkeypatch: Generator[MonkeyPatch, None, None],
    tmp_path: Path,
    model_class: Module,
    sensor: str,
    bands: str,
    in_channels: int,
    num_classes: int,
) -> None:
    extract_archive(
        os.path.join("tests", "data", "models", "resnet50-sentinel2-2.pt.zip"),
        str(tmp_path),
    )

    new_model_urls = {
        "sentinel2": {"all": {"resnet50": str(tmp_path / "resnet50-sentinel2.pt")}}
    }

    monkeypatch.setattr(  # type: ignore[attr-defined]
        torchgeo.models.resnet, "MODEL_URLS", new_model_urls
    )

    model = model_class(sensor, bands, pretrained=True)
    x = torch.zeros(1, in_channels, 256, 256)  # type: ignore[attr-defined]
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.size() == torch.Size([1, 17])  # type: ignore[attr-defined]
