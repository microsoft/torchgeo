# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch
from torch.nn.modules import Module

from torchgeo.models import resnet50


@pytest.mark.parametrize(
    "model_class,sensor,in_channels,num_classes", [(resnet50, "sentinel2", 10, 17)]
)
def test_resnet(
    model_class: Module, sensor: str, in_channels: int, num_classes: int
) -> None:
    model = model_class(sensor, pretrained=True)
    x = torch.zeros(1, in_channels, 256, 256)  # type: ignore[attr-defined]
    y = model(x)
    assert isinstance(y, torch.Tensor)
    assert y.size() == torch.Size([1, 17])  # type: ignore[attr-defined]
