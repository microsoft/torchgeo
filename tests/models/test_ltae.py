# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for LTAE model."""

import pytest
import torch

from torchgeo.models import LTAE


class TestLTAE:
    """Tests for LTAE model."""

    def test_forward(self) -> None:
        """Test forward pass."""
        batch_size = 4
        seq_len = 24
        in_channels = 128

        model = LTAE(in_channels=in_channels)
        x = torch.randn(batch_size, seq_len, in_channels)
        output = model(x)

        assert output.shape[0] == (batch_size, model.n_neurons[-1])
        assert len(output.shape) == 2  # (batch_size, embedding_dim)

    @pytest.mark.parametrize('in_channels', [64, 128, 256])
    def test_input_channels(self, in_channels: int) -> None:
        """Test different input channel configurations."""
        batch_size = 4
        seq_len = 24

        model = LTAE(in_channels=in_channels)
        x = torch.randn(batch_size, seq_len, in_channels)
        output = model(x)

        assert output.shape[0] == batch_size

    def test_invalid_input(self) -> None:
        """Test invalid input shape."""
        batch_size = 4
        seq_len = 24
        in_channels = 128
        wrong_channels = 64  # Different from model's in_channels

        with pytest.raises(AssertionError):
            model = LTAE(in_channels=in_channels)
            x = torch.randn(batch_size, seq_len, wrong_channels)
            model(x)
