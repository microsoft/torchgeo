# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for LTAE models."""

import pytest
import torch

from torchgeo.models import LTAE, LTAE2d


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

        assert output.shape == (batch_size, model.n_neurons[-1])
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

        with pytest.raises(RuntimeError):
            model = LTAE(in_channels=in_channels)
            x = torch.randn(batch_size, seq_len, wrong_channels)
            model(x)


class TestLTAE2d:
    """Tests for the LTAE2d model."""

    def test_forward(self) -> None:
        """Basic forward pass without positional encoding."""
        model = LTAE2d(
            in_channels=32,
            n_head=4,
            d_model=32,
            mlp=(32, 16),
            d_k=4,
            positional_encoding=False,
        )
        x = torch.randn(2, 5, 32, 8, 8)
        out = model(x)
        assert out.shape == (2, 16, 8, 8)

    def test_forward_with_positions(self) -> None:
        """Forward pass with acquisition-date positional encoding."""
        model = LTAE2d(in_channels=32, n_head=4, d_model=32, mlp=(32, 16), d_k=4)
        x = torch.randn(2, 5, 32, 8, 8)
        positions = torch.randint(1, 365, (2, 5))
        out = model(x, batch_positions=positions)
        assert out.shape == (2, 16, 8, 8)

    def test_return_att(self) -> None:
        """Return attention masks when return_att=True."""
        model = LTAE2d(
            in_channels=32,
            n_head=4,
            d_model=32,
            mlp=(32, 16),
            d_k=4,
            return_att=True,
            positional_encoding=False,
        )
        x = torch.randn(2, 5, 32, 8, 8)
        out, att = model(x)
        assert out.shape == (2, 16, 8, 8)
        assert att.shape == (4, 2, 5, 8, 8)  # (n_head, B, T, H, W)

    def test_pad_mask(self) -> None:
        """Forward pass with a padding mask."""
        model = LTAE2d(
            in_channels=32,
            n_head=4,
            d_model=32,
            mlp=(32, 16),
            d_k=4,
            return_att=True,
            positional_encoding=False,
        )
        x = torch.randn(2, 5, 32, 8, 8)
        pad_mask = torch.zeros(2, 5, dtype=torch.bool)
        pad_mask[0, -1] = True  # last timestep of first item is padded
        out, att = model(x, pad_mask=pad_mask)
        assert out.shape == (2, 16, 8, 8)
        assert att.shape == (4, 2, 5, 8, 8)

    def test_no_d_model(self) -> None:
        """Forward pass without an input projection (d_model=None)."""
        model = LTAE2d(
            in_channels=32,
            n_head=4,
            d_model=None,
            mlp=(32, 16),
            d_k=4,
            positional_encoding=False,
        )
        x = torch.randn(2, 5, 32, 8, 8)
        out = model(x)
        assert out.shape == (2, 16, 8, 8)

    def test_invalid_n_head(self) -> None:
        """Invalid number of attention heads raises a clear error."""
        with pytest.raises(ValueError, match='n_head must be positive'):
            LTAE2d(in_channels=32, n_head=0, d_model=32, mlp=(32, 16))

    def test_in_channels_must_be_divisible_by_n_head(self) -> None:
        """Input channels must be divisible by the number of attention heads."""
        match = 'in_channels must be divisible by n_head'
        with pytest.raises(ValueError, match=match):
            LTAE2d(in_channels=30, n_head=4, d_model=32, mlp=(32, 16))

    def test_d_model_must_be_divisible_by_n_head(self) -> None:
        """Attention projection width must be divisible by attention heads."""
        match = 'd_model must be divisible by n_head'
        with pytest.raises(ValueError, match=match):
            LTAE2d(in_channels=32, n_head=4, d_model=30, mlp=(30, 16))

    def test_mlp_output_must_be_divisible_by_n_head(self) -> None:
        """Output channels must be divisible by attention heads."""
        match = 'mlp\\[-1\\] must be divisible by n_head'
        with pytest.raises(ValueError, match=match):
            LTAE2d(in_channels=32, n_head=4, d_model=32, mlp=(32, 14))
