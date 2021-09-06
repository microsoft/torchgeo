# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import itertools

import pytest
import torch

from torchgeo.models import FCEF, FCSiamConc, FCSiamDiff

BATCH_SIZE = [1, 2]
CHANNELS = [1, 3, 5]
CLASSES = [2, 3]
T = [2, 3]


class TestFCEF:
    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, c", list(itertools.product(BATCH_SIZE, CHANNELS)))
    def test_in_channels(self, b: int, c: int) -> None:
        classes = 2
        t, h, w = 2, 64, 64
        model = FCEF(in_channels=c, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, classes", list(itertools.product(BATCH_SIZE, CLASSES)))
    def test_classes(self, b: int, classes: int) -> None:
        t, c, h, w = 2, 3, 64, 64
        model = FCEF(in_channels=3, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, t", list(itertools.product(BATCH_SIZE, T)))
    def test_t(self, b: int, t: int) -> None:
        classes = 2
        c, h, w = 3, 64, 64
        model = FCEF(in_channels=3, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)


class TestFCSiamConc:
    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, c", list(itertools.product(BATCH_SIZE, CHANNELS)))
    def test_in_channels(self, b: int, c: int) -> None:
        classes = 2
        t, h, w = 2, 64, 64
        model = FCSiamConc(in_channels=c, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, classes", list(itertools.product(BATCH_SIZE, CLASSES)))
    def test_classes(self, b: int, classes: int) -> None:
        t, c, h, w = 2, 3, 64, 64
        model = FCSiamConc(in_channels=3, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, t", list(itertools.product(BATCH_SIZE, T)))
    def test_t(self, b: int, t: int) -> None:
        classes = 2
        c, h, w = 3, 64, 64
        model = FCSiamConc(in_channels=3, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)


class TestFCSiamDiff:
    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, c", list(itertools.product(BATCH_SIZE, CHANNELS)))
    def test_in_channels(self, b: int, c: int) -> None:
        classes = 2
        t, h, w = 2, 64, 64
        model = FCSiamDiff(in_channels=c, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, classes", list(itertools.product(BATCH_SIZE, CLASSES)))
    def test_classes(self, b: int, classes: int) -> None:
        t, c, h, w = 2, 3, 64, 64
        model = FCSiamDiff(in_channels=3, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)

    @torch.no_grad()  # type: ignore[misc]
    @pytest.mark.parametrize("b, t", list(itertools.product(BATCH_SIZE, T)))
    def test_t(self, b: int, t: int) -> None:
        classes = 2
        c, h, w = 3, 64, 64
        model = FCSiamDiff(in_channels=3, t=t, classes=classes)
        x = torch.randn(b, t, c, h, w)
        y = model(x)
        assert y.shape == (b, classes, h, w)
