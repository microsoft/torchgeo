# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for U-TAE models."""

import pytest
import torch

from torchgeo.models import UTAE
from torchgeo.models.utae import ConvBlock, ConvLayer, TemporalAggregator


def _first_batch_norm(module: torch.nn.Module) -> torch.nn.BatchNorm2d:
    """Return the first BatchNorm2d module."""
    for child in module.modules():
        if isinstance(child, torch.nn.BatchNorm2d):
            return child

    msg = 'Expected module to contain BatchNorm2d'
    raise AssertionError(msg)


def _assert_batch_norm_state_equal(
    actual: torch.nn.BatchNorm2d, expected: torch.nn.BatchNorm2d
) -> None:
    """Assert that two BatchNorm2d running states are equal."""
    assert actual.running_mean is not None
    assert expected.running_mean is not None
    assert actual.running_var is not None
    assert expected.running_var is not None
    assert actual.num_batches_tracked is not None
    assert expected.num_batches_tracked is not None
    assert torch.allclose(actual.running_mean, expected.running_mean)
    assert torch.allclose(actual.running_var, expected.running_var)
    assert torch.equal(actual.num_batches_tracked, expected.num_batches_tracked)


class TestUTAE:
    """Tests for the UTAE model."""

    @pytest.fixture
    def small_model(self) -> UTAE:
        """Small UTAE for fast testing."""
        return UTAE(
            input_dim=4,
            encoder_widths=(32, 64),
            decoder_widths=(16, 64),
            out_conv=(8, 3),
            n_head=16,
            d_model=64,
            d_k=4,
        )

    @pytest.fixture
    def x(self) -> torch.Tensor:
        """Batch of image time series (B=2, T=4, C=4, H=16, W=16)."""
        return torch.randn(2, 4, 4, 16, 16)

    def test_forward(self, small_model: UTAE, x: torch.Tensor) -> None:
        """Basic forward pass."""
        out = small_model(x)
        assert out.shape == (2, 3, 16, 16)

    def test_return_att(self, small_model: UTAE, x: torch.Tensor) -> None:
        """return_att=True yields output and attention masks."""
        out, att = small_model(x, return_att=True)
        assert out.shape == (2, 3, 16, 16)
        assert att.shape[0] == 16  # n_head

    def test_return_maps(self, x: torch.Tensor) -> None:
        """return_maps=True yields output and feature map list."""
        model = UTAE(
            input_dim=4,
            encoder_widths=(32, 64),
            decoder_widths=(16, 64),
            out_conv=(8, 3),
            n_head=16,
            d_model=64,
            d_k=4,
            return_maps=True,
        )
        out, maps = model(x)
        assert out.shape == (2, 3, 16, 16)
        assert isinstance(maps, list)
        assert len(maps) > 0

    def test_encoder_mode(self, x: torch.Tensor) -> None:
        """encoder=True returns feature maps instead of class scores."""
        model = UTAE(
            input_dim=4,
            encoder_widths=(32, 64),
            decoder_widths=(16, 64),
            out_conv=(8, 3),
            n_head=16,
            d_model=64,
            d_k=4,
            encoder=True,
        )
        _, maps = model(x)
        assert isinstance(maps, list)
        assert len(maps) > 0

    @pytest.mark.parametrize('agg_mode', ['att_group', 'att_mean', 'mean'])
    def test_agg_modes(self, x: torch.Tensor, agg_mode: str) -> None:
        """All skip-connection aggregation modes produce correct output shapes."""
        model = UTAE(
            input_dim=4,
            encoder_widths=(32, 64),
            decoder_widths=(16, 64),
            out_conv=(8, 3),
            n_head=16,
            d_model=64,
            d_k=4,
            agg_mode=agg_mode,
        )
        out = model(x)
        assert out.shape == (2, 3, 16, 16)

    def test_decoder_widths_none(self, x: torch.Tensor) -> None:
        """decoder_widths=None mirrors encoder widths."""
        model = UTAE(
            input_dim=4,
            encoder_widths=(32, 32),
            decoder_widths=None,
            out_conv=(8, 3),
            n_head=16,
            d_model=32,
            d_k=4,
        )
        out = model(x)
        assert out.shape == (2, 3, 16, 16)

    def test_width_lengths_must_match(self) -> None:
        """Encoder and decoder widths must have the same number of stages."""
        match = 'encoder_widths and decoder_widths must have the same length'

        with pytest.raises(ValueError, match=match):
            UTAE(input_dim=4, encoder_widths=(32, 64), decoder_widths=(16, 32, 64))

    def test_final_widths_must_match(self) -> None:
        """Encoder and decoder bottleneck widths must match."""
        match = 'encoder_widths and decoder_widths must have the same final width'

        with pytest.raises(ValueError, match=match):
            UTAE(input_dim=4, encoder_widths=(32, 64), decoder_widths=(16, 32))

    def test_forward_with_positions(self, small_model: UTAE, x: torch.Tensor) -> None:
        """batch_positions triggers the date-based positional encoder in L-TAE 2D."""
        batch_positions = torch.randint(1, 366, (2, 4))
        out = small_model(x, batch_positions=batch_positions)
        assert out.shape == (2, 3, 16, 16)

    @pytest.mark.parametrize('agg_mode', ['att_group', 'att_mean', 'mean'])
    def test_forward_with_padding(self, agg_mode: str) -> None:
        """All-zero frames (matching pad_value=0) exercise the padding mask paths."""
        model = UTAE(
            input_dim=4,
            encoder_widths=(32, 64),
            decoder_widths=(16, 64),
            out_conv=(8, 3),
            n_head=16,
            d_model=64,
            d_k=4,
            agg_mode=agg_mode,
        )
        x = torch.randn(2, 4, 4, 16, 16)
        x[:, 2:] = 0.0  # last two timesteps are padding
        out = model(x)
        assert out.shape == (2, 3, 16, 16)

    @pytest.mark.parametrize('encoder_norm', ['instance', 'none'])
    def test_encoder_norm(self, x: torch.Tensor, encoder_norm: str) -> None:
        """Instance norm and no-norm branches in ConvLayer."""
        model = UTAE(
            input_dim=4,
            encoder_widths=(32, 64),
            decoder_widths=(16, 64),
            out_conv=(8, 3),
            n_head=16,
            d_model=64,
            d_k=4,
            encoder_norm=encoder_norm,
        )
        out = model(x)
        assert out.shape == (2, 3, 16, 16)

    def test_conv_layer_last_relu_false_keeps_intermediate_relu(self) -> None:
        """last_relu=False omits only the final ReLU."""
        layer = ConvLayer(nkernels=(1, 2, 3), norm='none', last_relu=False)

        relus = [module for module in layer.conv if isinstance(module, torch.nn.ReLU)]

        assert len(relus) == 1
        assert isinstance(layer.conv[1], torch.nn.ReLU)
        assert not isinstance(layer.conv[-1], torch.nn.ReLU)

    def test_temporal_aggregator_mean_all_padded_returns_zero(self) -> None:
        """Mean aggregation handles all-padded sequences without NaNs."""
        aggregator = TemporalAggregator(mode='mean')
        x = torch.randn(2, 3, 4, 5, 5)
        pad_mask = torch.tensor(
            [[True, True, True], [False, True, True]], dtype=torch.bool
        )

        out = aggregator(x, pad_mask=pad_mask)

        assert torch.isfinite(out).all()
        assert torch.all(out[0] == 0)
        assert torch.allclose(out[1], x[1, 0])

    def test_temporal_aggregator_att_group_requires_divisible_channels(self) -> None:
        """att_group raises a clear error when heads do not divide channels."""
        aggregator = TemporalAggregator(mode='att_group')
        x = torch.randn(2, 3, 5, 4, 4)
        attn_mask = torch.rand(2, 2, 3, 4, 4)
        match = 'x.shape\\[2\\] must be divisible by n_heads'

        with pytest.raises(ValueError, match=match):
            aggregator(x, attn_mask=attn_mask)

    def test_smart_forward_without_pad_value(self) -> None:
        """smart_forward applies the block when pad_value is None."""
        block = ConvBlock(nkernels=(1, 2), pad_value=None, norm='none')
        x = torch.randn(2, 3, 1, 8, 8)

        expected = block.forward(x.view(6, 1, 8, 8)).view(2, 3, 2, 8, 8)
        actual = block.smart_forward(x)

        assert torch.allclose(actual, expected)

    def test_smart_forward_rejects_four_dimensional_input(self) -> None:
        """smart_forward only supports temporal 5-D inputs."""
        block = ConvBlock(nkernels=(1, 2), pad_value=None, norm='none')
        x = torch.randn(3, 1, 8, 8)

        with pytest.raises(ValueError, match='x must have shape'):
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

        batch_norm = _first_batch_norm(block)
        expected_batch_norm = _first_batch_norm(expected_block)
        _assert_batch_norm_state_equal(batch_norm, expected_batch_norm)

    def test_smart_forward_all_padded_preserves_batch_norm_stats(self) -> None:
        """All-padded fallback infers shape without mutating BatchNorm state."""
        block = ConvBlock(nkernels=(1, 1), pad_value=0, norm='batch')
        batch_norm = _first_batch_norm(block)
        assert batch_norm.running_mean is not None
        assert batch_norm.running_var is not None
        assert batch_norm.num_batches_tracked is not None
        running_mean = batch_norm.running_mean.clone()
        running_var = batch_norm.running_var.clone()
        num_batches_tracked = batch_norm.num_batches_tracked.clone()
        x = torch.zeros(1, 2, 1, 8, 8)

        block.train()
        out = block.smart_forward(x)

        assert out.shape == (1, 2, 1, 8, 8)
        assert torch.all(out == 0)
        assert batch_norm.running_mean is not None
        assert batch_norm.running_var is not None
        assert batch_norm.num_batches_tracked is not None
        assert torch.equal(batch_norm.running_mean, running_mean)
        assert torch.equal(batch_norm.running_var, running_var)
        assert torch.equal(batch_norm.num_batches_tracked, num_batches_tracked)
