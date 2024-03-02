# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch

from torchgeo.datasets import EuroSAT
from torchgeo.models import RCF


class TestRCF:
    def test_in_channels(self) -> None:
        model = RCF(in_channels=5, features=4, kernel_size=3, mode="gaussian")
        x = torch.randn(2, 5, 64, 64)
        model(x)

        model = RCF(in_channels=3, features=4, kernel_size=3, mode="gaussian")
        match = "to have 3 channels, but got 5 channels instead"
        with pytest.raises(RuntimeError, match=match):
            model(x)

    def test_num_features(self) -> None:
        model = RCF(in_channels=5, features=4, kernel_size=3, mode="gaussian")
        x = torch.randn(2, 5, 64, 64)
        y = model(x)
        assert y.shape[1] == 4

        x = torch.randn(1, 5, 64, 64)
        y = model(x)
        assert y.shape[0] == 4

    def test_untrainable(self) -> None:
        model = RCF(in_channels=5, features=4, kernel_size=3, mode="gaussian")
        assert len(list(model.parameters())) == 0

    def test_biases(self) -> None:
        model = RCF(features=24, bias=10, mode="gaussian")
        # https://github.com/pytorch/pytorch/issues/116328
        assert torch.all(model.biases == 10)

    def test_seed(self) -> None:
        weights1 = RCF(seed=1, mode="gaussian").weights
        weights2 = RCF(seed=1, mode="gaussian").weights
        assert torch.allclose(weights1, weights2)

    def test_empirical(self) -> None:
        root = os.path.join("tests", "data", "eurosat")
        ds = EuroSAT(root=root, bands=EuroSAT.rgb_bands, split="train")
        model = RCF(
            in_channels=3, features=4, kernel_size=3, mode="empirical", dataset=ds
        )
        model(torch.randn(2, 3, 8, 8))

    def test_empirical_no_dataset(self) -> None:
        match = "dataset must be provided when mode is 'empirical'"
        with pytest.raises(ValueError, match=match):
            RCF(mode="empirical", dataset=None)
