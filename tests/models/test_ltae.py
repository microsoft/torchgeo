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

    @pytest.mark.parametrize(
        ('d_model', 'positional_encoding', 'mode'),
        [
            pytest.param(32, False, 'basic', id='basic'),
            pytest.param(32, True, 'positions', id='positions'),
            pytest.param(None, False, 'basic', id='no-projection'),
            pytest.param(32, False, 'padding', id='padding'),
        ],
    )
    def test_forward(
        self, d_model: int | None, positional_encoding: bool, mode: str
    ) -> None:
        """Test forward-pass variants and attention masks."""
        model = LTAE2d(
            in_channels=32,
            n_head=4,
            d_model=d_model,
            mlp=(32, 16),
            d_k=4,
            positional_encoding=positional_encoding,
        )
        x = torch.randn(2, 5, 32, 8, 8)
        positions = torch.randint(1, 365, (2, 5)) if mode == 'positions' else None
        pad_mask = torch.zeros(2, 5, dtype=torch.bool) if mode == 'padding' else None
        if pad_mask is not None:
            pad_mask[0, -1] = True
        out, att = model(x, batch_positions=positions, pad_mask=pad_mask)
        assert out.shape == (2, 16, 8, 8)
        assert att.shape == (4, 2, 5, 8, 8)

    @pytest.mark.parametrize(
        ('in_channels', 'n_head', 'd_model', 'mlp', 'error', 'match'),
        [
            pytest.param(
                30, 4, 32, (32, 16), AssertionError, 'in_channels', id='channels'
            ),
            pytest.param(32, 4, 30, (30, 16), AssertionError, 'd_model', id='d-model'),
            pytest.param(32, 4, 32, (32, 14), ValueError, 'mlp', id='mlp'),
        ],
    )
    def test_invalid_config(
        self,
        in_channels: int,
        n_head: int,
        d_model: int,
        mlp: tuple[int, int],
        error: type[Exception],
        match: str,
    ) -> None:
        """Test invalid model configurations."""
        with pytest.raises(error, match=match):
            LTAE2d(in_channels=in_channels, n_head=n_head, d_model=d_model, mlp=mlp)
