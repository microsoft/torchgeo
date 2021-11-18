# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import RCF


class TestRCF:
    def test_in_channels(self) -> None:
        model = RCF(in_channels=5, features=4, kernel_size=3)
        x = torch.randn(2, 5, 64, 64)
        model(x)

        model = RCF(in_channels=3, features=4, kernel_size=3)
        match = "to have 3 channels, but got 5 channels instead"
        with pytest.raises(RuntimeError, match=match):
            model(x)

    def test_num_features(self) -> None:
        model = RCF(in_channels=5, features=4, kernel_size=3)
        x = torch.randn(2, 5, 64, 64)
        y = model(x)
        assert y.shape[1] == 4

        x = torch.randn(1, 5, 64, 64)
        y = model(x)
        assert y.shape[0] == 4

    def test_untrainable(self) -> None:
        model = RCF(in_channels=5, features=4, kernel_size=3)
        assert len(list(model.parameters())) == 0

    def test_biases(self) -> None:
        model = RCF(features=24, bias=10)
        assert torch.all(model.biases == 10)  # type: ignore[attr-defined]

    def test_seed(self) -> None:
        weights1 = RCF(seed=1).weights
        weights2 = RCF(seed=1).weights
        assert torch.allclose(weights1, weights2)  # type: ignore[attr-defined]
