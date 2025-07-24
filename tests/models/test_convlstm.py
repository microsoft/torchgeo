# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Tests for the ConvLSTM model."""

import pytest
import torch

from torchgeo.models import ConvLSTM


class TestConvLSTM:
    """Tests for the ConvLSTM model."""

    def test_convlstm_forward(self) -> None:
        """Test the forward pass of the ConvLSTM model."""
        b = 1
        t = 4
        c = 3
        h = 64
        w = 64
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(
            input_dim=c,
            hidden_dim=16,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=True,
        )
        layer_output_list, last_state_list = model(input_tensor)

        assert len(layer_output_list) == 1
        assert len(last_state_list) == 1
        assert layer_output_list[0].shape == (b, t, 16, h, w)

    def test_convlstm_multilayers(self) -> None:
        """Test the forward pass with multiple layers."""
        b = 1
        t = 4
        c = 3
        h = 64
        w = 64
        hidden_dims = [16, 32]
        num_layers = 2
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(
            input_dim=c,
            hidden_dim=hidden_dims,
            kernel_size=(3, 3),
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=True,
        )
        layer_output_list, _ = model(input_tensor)

        assert len(layer_output_list) == num_layers
        assert layer_output_list[0].shape == (b, t, hidden_dims[0], h, w)
        assert layer_output_list[1].shape == (b, t, hidden_dims[1], h, w)

    def test_convlstm_invalid_kernel_size(self) -> None:
        """Test that an invalid kernel size raises a ValueError."""
        with pytest.raises(ValueError):
            ConvLSTM(input_dim=3, hidden_dim=16, kernel_size=[(3)], num_layers=1)
