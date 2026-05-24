# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the ConvLSTM model."""

from typing import Literal, cast

import pytest
import torch

from torchgeo.models import Conv3dLSTM, ConvLSTM


class TestConvLSTM:
    """Tests for the ConvLSTM model."""

    def test_convlstm_forward_features(self) -> None:
        """Test the feature forward pass of the ConvLSTM model."""
        b = 1
        t = 4
        c = 3
        h = 64
        w = 64
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(input_dim=c, hidden_dim=16, kernel_size=(3, 3), num_layers=1)
        layer_output_list, last_state_list = model.forward_features(input_tensor)

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
            return_all_layers=True,
        )
        layer_output_list, _ = model.forward_features(input_tensor)

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
        )
        layer_output_list, last_state_list = model.forward_features(input_tensor)

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
        )
        layer_output_list, last_state_list = model.forward_features(input_tensor)

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
            return_all_layers=True,
        )
        layer_output_list, last_state_list = model.forward_features(input_tensor)

        assert len(layer_output_list) == 2
        assert len(last_state_list) == 2
        assert layer_output_list[0].shape == (b, t, 16, h, w)
        assert layer_output_list[1].shape == (b, t, 32, h, w)

    def test_convlstm_forward(self) -> None:
        """Test segmentation forward pass with prediction head."""
        b = 2
        t = 4
        c = 3
        h = 16
        w = 16
        num_classes = 5
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(
            input_dim=c,
            hidden_dim=16,
            kernel_size=3,
            num_layers=1,
            num_classes=num_classes,
            head_kernel_size=1,
            convolutional_head=True,
        )
        y_hat = model(input_tensor, lengths=torch.tensor([4, 2]))

        assert y_hat.shape == (b, num_classes, h, w)

    def test_convlstm_forward_uses_last_timestep_without_lengths(self) -> None:
        """Test segmentation forward pass defaults to the final timestep."""
        b = 2
        t = 4
        c = 3
        h = 16
        w = 16
        num_classes = 5
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(
            input_dim=c,
            hidden_dim=16,
            kernel_size=3,
            num_layers=1,
            num_classes=num_classes,
            head_kernel_size=1,
        )
        y_hat = model(input_tensor)
        y_hat_last_step = model(input_tensor, lengths=torch.tensor([t, t]))

        torch.testing.assert_close(y_hat, y_hat_last_step)

    def test_convlstm_forward_clamps_lengths_exceeding_sequence(self) -> None:
        """Test that lengths longer than the sequence clamp to the final timestep."""
        b = 2
        t = 4
        c = 3
        h = 16
        w = 16
        input_tensor = torch.rand(b, t, c, h, w)

        model = ConvLSTM(
            input_dim=c, hidden_dim=16, kernel_size=3, num_layers=1, num_classes=5
        )
        y_hat_clamped = model(input_tensor, lengths=torch.tensor([9.0, 12.0]))
        y_hat_last = model(input_tensor)

        torch.testing.assert_close(y_hat_clamped, y_hat_last)


class TestConv3dLSTM:
    """Tests for the Conv3dLSTM model."""

    @pytest.mark.parametrize(
        ('output_mode', 'return_sequence', 'pooling', 'lengths', 'expected_shape'),
        [
            ('pixel', False, 'avg', None, (2, 5, 16, 16)),
            ('pixel', True, 'avg', torch.tensor([4, 2]), (2, 4, 5, 16, 16)),
            ('chip', False, 'avg', torch.tensor([4, 2]), (2, 5)),
            ('chip', True, 'max', torch.tensor([4, 2]), (2, 4, 5)),
        ],
    )
    def test_conv3dlstm_forward(
        self,
        output_mode: Literal['pixel', 'chip'],
        return_sequence: bool,
        pooling: Literal['avg', 'max'],
        lengths: torch.Tensor | None,
        expected_shape: tuple[int, ...],
    ) -> None:
        """Test forward pass output shapes."""
        b = 2
        t = 4
        c = 3
        h = 16
        w = 16
        num_outputs = 5
        input_tensor = torch.rand(b, t, c, h, w)

        model = Conv3dLSTM(
            input_dim=c,
            conv3d_dim=8,
            hidden_dim=16,
            num_outputs=num_outputs,
            output_mode=output_mode,
            return_sequence=return_sequence,
            pooling=pooling,
        )
        y_hat = model(input_tensor, lengths=lengths)

        assert y_hat.shape == expected_shape

    def test_conv3dlstm_forward_features(self) -> None:
        """Test feature forward pass."""
        b = 2
        t = 4
        c = 3
        h = 16
        w = 16
        input_tensor = torch.rand(b, t, c, h, w)

        model = Conv3dLSTM(input_dim=c, conv3d_dim=8, hidden_dim=16)
        layer_output_list, last_state_list = model.forward_features(input_tensor)

        assert len(layer_output_list) == 1
        assert len(last_state_list) == 1
        assert layer_output_list[0].shape == (b, t, 16, h, w)

    def test_conv3dlstm_conv3d_kernel_size_as_int(self) -> None:
        """Test that conv3d_kernel_size can be an integer."""
        model = Conv3dLSTM(input_dim=3, conv3d_kernel_size=3)

        assert model.conv3d_kernel_size == (3, 3, 3)

    @pytest.mark.parametrize(
        ('option', 'match'),
        [
            ('conv3d_kernel_size', 'conv3d_kernel_size must be odd'),
            ('head_kernel_size', 'head_kernel_size must be odd'),
            ('output_mode', "output_mode must be 'pixel' or 'chip'"),
            ('pooling', "pooling must be 'avg' or 'max'"),
        ],
    )
    def test_conv3dlstm_invalid_init(
        self,
        option: Literal[
            'conv3d_kernel_size', 'head_kernel_size', 'output_mode', 'pooling'
        ],
        match: str,
    ) -> None:
        """Test that invalid initialization options raise a ValueError."""
        with pytest.raises(ValueError, match=match):
            match option:
                case 'conv3d_kernel_size':
                    Conv3dLSTM(input_dim=3, conv3d_kernel_size=(3, 2, 3))
                case 'head_kernel_size':
                    Conv3dLSTM(input_dim=3, head_kernel_size=2)
                case 'output_mode':
                    output_mode = cast('Literal["pixel", "chip"]', 'invalid')
                    Conv3dLSTM(input_dim=3, output_mode=output_mode)
                case 'pooling':
                    pooling = cast('Literal["avg", "max"]', 'invalid')
                    Conv3dLSTM(input_dim=3, pooling=pooling)

    def test_conv3dlstm_invalid_input_shape(self) -> None:
        """Test that invalid input shape raises a ValueError."""
        model = Conv3dLSTM(input_dim=3)
        input_tensor = torch.rand(2, 3, 16, 16)

        with pytest.raises(ValueError, match='Expected input_tensor'):
            model(input_tensor)
