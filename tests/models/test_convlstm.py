# Copyright (c) TorchGeo Contributors. All rights reserved.
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

    def test_convlstm_kernel_size_as_int(self) -> None:
        """Test that kernel_size can be an integer."""
        b = 1
        t = 4
        c = 3
        h = 64
        w = 64
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(
            input_dim=c,
            hidden_dim=16,
            kernel_size=3,  # Pass as integer
            num_layers=1,
            batch_first=True,
        )
        layer_output_list, last_state_list = model(input_tensor)

        assert len(layer_output_list) == 1
        assert len(last_state_list) == 1
        assert layer_output_list[0].shape == (b, t, 16, h, w)

    def test_convlstm_kernel_size_as_list(self) -> None:
        """Test that kernel_size can be a list of tuples."""
        b = 1
        t = 4
        c = 3
        h = 64
        w = 64
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(
            input_dim=c,
            hidden_dim=16,
            kernel_size=[(3, 3)],  # Pass as list of tuples
            num_layers=1,
            batch_first=True,
        )
        layer_output_list, last_state_list = model(input_tensor)

        assert len(layer_output_list) == 1
        assert len(last_state_list) == 1
        assert layer_output_list[0].shape == (b, t, 16, h, w)

    def test_convlstm_batch_first_false(self) -> None:
        """Test the forward pass with batch_first=False."""
        b = 1
        t = 4
        c = 3
        h = 64
        w = 64
        input_tensor = torch.rand(t, b, c, h, w)  # Note the different order

        model = ConvLSTM(
            input_dim=c,
            hidden_dim=16,
            kernel_size=(3, 3),
            num_layers=1,
            batch_first=False,
        )
        layer_output_list, last_state_list = model(input_tensor)

        assert len(layer_output_list) == 1
        assert len(last_state_list) == 1
        assert layer_output_list[0].shape == (b, t, 16, h, w)

    def test_convlstm_inconsistent_list_length(self) -> None:
        """Test that inconsistent list lengths raise a ValueError."""
        with pytest.raises(ValueError, match='Inconsistent list length'):
            ConvLSTM(
                input_dim=3,
                hidden_dim=[16, 32],  # 2 layers
                kernel_size=[(3, 3)],  # 1 layer
                num_layers=2,
            )

    def test_convlstm_mixed_kernel_sizes(self) -> None:
        """Test that kernel_size can be a list of mixed ints and tuples."""
        b = 1
        t = 4
        c = 3
        h = 64
        w = 64
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(
            input_dim=c,
            hidden_dim=[16, 32],
            kernel_size=[3, (5, 5)],  # Mix of int and tuple
            num_layers=2,
            batch_first=True,
            return_all_layers=True,
        )
        layer_output_list, last_state_list = model(input_tensor)

        assert len(layer_output_list) == 2
        assert len(last_state_list) == 2
        assert layer_output_list[0].shape == (b, t, 16, h, w)
        assert layer_output_list[1].shape == (b, t, 32, h, w)
