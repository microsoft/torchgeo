# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for U-TAE models."""

from typing import Any

import pytest
import torch

from torchgeo.models import UTAE
from torchgeo.models.utae import ConvBlock, ConvLayer, TemporalAggregator


def create_model(**kwargs: Any) -> UTAE:
    """Create a small UTAE model for fast testing."""
    config: dict[str, Any] = {
        'input_dim': 4,
        'encoder_widths': (32, 64),
        'decoder_widths': (16, 64),
        'out_conv': (8, 3),
        'n_head': 16,
        'd_model': 64,
        'd_k': 4,
    }
    config.update(kwargs)
    return UTAE(**config)


class TestUTAE:
    """Tests for the UTAE model."""

    @pytest.fixture
    def x(self) -> torch.Tensor:
        """Batch of image time series (B=2, T=4, C=4, H=16, W=16)."""
        return torch.randn(2, 4, 4, 16, 16)

    @pytest.mark.parametrize(
        ('kwargs', 'padded'),
        [
            pytest.param({}, False, id='basic'),
            pytest.param(
                {'encoder_widths': (32, 32), 'decoder_widths': None, 'd_model': 32},
                False,
                id='mirrored-decoder',
            ),
            pytest.param({'encoder_norm': 'instance'}, False, id='instance-norm'),
            pytest.param({'encoder_norm': 'none'}, False, id='no-norm'),
            pytest.param({}, True, id='padding'),
        ],
    )
    def test_forward(
        self, x: torch.Tensor, kwargs: dict[str, Any], padded: bool
    ) -> None:
        """Test forward-pass configurations."""
        if padded:
            x[:, 2:] = 0
        out = create_model(**kwargs)(x)
        assert out.shape == (2, 3, 16, 16)

    def test_return_att(self, x: torch.Tensor) -> None:
        """return_att=True yields output and attention masks."""
        out, att = create_model()(x, return_att=True)
        assert out.shape == (2, 3, 16, 16)
        assert att.shape[0] == 16  # n_head

    @pytest.mark.parametrize('encoder', [False, True])
    def test_return_maps(self, x: torch.Tensor, encoder: bool) -> None:
        """Return decoder feature maps in output and encoder modes."""
        _, maps = create_model(return_maps=True, encoder=encoder)(x)
        assert isinstance(maps, list)
        assert len(maps) > 0

    @pytest.mark.parametrize(
        ('decoder_widths', 'match'),
        [
            pytest.param((16, 32, 64), 'same length', id='length'),
            pytest.param((16, 32), 'same final width', id='final-width'),
        ],
    )
    def test_invalid_widths(self, decoder_widths: tuple[int, ...], match: str) -> None:
        """Test incompatible encoder and decoder widths."""
        with pytest.raises(AssertionError, match=match):
            create_model(decoder_widths=decoder_widths)

    def test_conv_layer_last_relu_false_keeps_intermediate_relu(self) -> None:
        """last_relu=False omits only the final ReLU."""
        layer = ConvLayer(nkernels=(1, 2, 3), norm='none', last_relu=False)

        relus = [module for module in layer.conv if isinstance(module, torch.nn.ReLU)]

        assert len(relus) == 1
        assert isinstance(layer.conv[1], torch.nn.ReLU)
        assert not isinstance(layer.conv[-1], torch.nn.ReLU)

    def test_temporal_aggregator_all_padded_returns_zero(self) -> None:
        """Attention-group aggregation handles all-padded sequences without NaNs."""
        aggregator = TemporalAggregator()
        x = torch.randn(2, 3, 4, 5, 5)
        attn_mask = torch.ones(2, 2, 3, 5, 5)
        pad_mask = torch.tensor(
            [[True, True, True], [False, True, True]], dtype=torch.bool
        )

        out = aggregator(x, pad_mask=pad_mask, attn_mask=attn_mask)

        assert torch.isfinite(out).all()
        assert torch.all(out[0] == 0)
        assert torch.allclose(out[1], x[1, 0])

    @pytest.mark.parametrize('mode', ['channels', 'attention'])
    def test_temporal_aggregator_invalid(self, mode: str) -> None:
        """Test invalid temporal aggregation inputs."""
        aggregator = TemporalAggregator()
        channels = 5 if mode == 'channels' else 4
        x = torch.randn(2, 3, channels, 4, 4)
        attn_mask = torch.rand(2, 2, 3, 4, 4) if mode == 'channels' else None
        match = 'divisible by n_heads' if mode == 'channels' else 'attn_mask'
        with pytest.raises(ValueError, match=match):
            aggregator(x, attn_mask=attn_mask)

    def test_smart_forward_without_pad_value(self) -> None:
        """smart_forward applies the block when pad_value is None."""
        block = ConvBlock(nkernels=(1, 2), pad_value=None, norm='none')
        x = torch.randn(2, 3, 1, 8, 8)

        expected = block.forward(x.view(6, 1, 8, 8)).view(2, 3, 2, 8, 8)
        actual = block.smart_forward(x)

        assert torch.allclose(actual, expected)

    @pytest.mark.parametrize('mode', ['dimensions', 'padding'])
    def test_smart_forward_invalid(self, mode: str) -> None:
        """Test invalid temporal block inputs."""
        block = ConvBlock(nkernels=(1, 2), pad_value=0, norm='none')
        x = (
            torch.randn(3, 1, 8, 8)
            if mode == 'dimensions'
            else torch.zeros(1, 2, 1, 8, 8)
        )
        match = (
            r'expected \(B, T, C, H, W\)' if mode == 'dimensions' else 'no valid frames'
        )
        with pytest.raises(ValueError, match=match):
            block.smart_forward(x)

    def test_smart_forward_skips_padded_batch_norm_stats(self) -> None:
        """Padded frames do not contribute to BatchNorm running stats."""
        block = ConvBlock(nkernels=(1, 1), pad_value=0, norm='batch')
        expected_block = ConvBlock(nkernels=(1, 1), pad_value=0, norm='batch')
        expected_block.load_state_dict(block.state_dict())
        x = torch.randn(1, 2, 1, 8, 8)
        x[:, 1] = 0

        block.train()
        expected_block.train()
        block.smart_forward(x)
        expected_block.forward(x[:, 0])

        batch_norm = next(
            module
            for module in block.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        )
        expected_batch_norm = next(
            module
            for module in expected_block.modules()
            if isinstance(module, torch.nn.BatchNorm2d)
        )
        for actual, expected in zip(
            batch_norm.buffers(), expected_batch_norm.buffers()
        ):
            assert torch.equal(actual, expected)
