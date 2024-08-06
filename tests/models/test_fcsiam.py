# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import FCSiamConc, FCSiamDiff

BATCH_SIZE = [1, 2]
CHANNELS = [1, 3, 5]
CLASSES = [1, 2]


class TestFCSiamConc:
    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    @pytest.mark.parametrize('c', CHANNELS)
    def test_in_channels(self, b: int, c: int) -> None:
        classes = 2
        t, h, w = 2, 64, 64
        model = FCSiamConc(in_channels=c, classes=classes, encoder_weights=None)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)

    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    @pytest.mark.parametrize('classes', CLASSES)
    def test_classes(self, b: int, classes: int) -> None:
        t, c, h, w = 2, 3, 64, 64
        model = FCSiamConc(in_channels=3, classes=classes, encoder_weights=None)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)


class TestFCSiamDiff:
    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    @pytest.mark.parametrize('c', CHANNELS)
    def test_in_channels(self, b: int, c: int) -> None:
        classes = 2
        t, h, w = 2, 64, 64
        model = FCSiamDiff(in_channels=c, classes=classes, encoder_weights=None)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)

    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    @pytest.mark.parametrize('classes', CLASSES)
    def test_classes(self, b: int, classes: int) -> None:
        t, c, h, w = 2, 3, 64, 64
        model = FCSiamDiff(in_channels=3, classes=classes, encoder_weights=None)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)
