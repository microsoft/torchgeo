# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for U-TAE models."""

import pytest
import torch

from torchgeo.models import UTAE


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
