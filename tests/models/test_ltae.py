"""
Unit tests for LTAE model
"""

import pytest
import torch

from torchgeo.models import LTAE


class TestLTAE:
    """Tests for the LTAE model."""

    def test_forward(self) -> None:
        """Test forward pass."""
        batch_size = 4
        seq_len = 24
        in_channels = 128
        
        model = LTAE(in_channels=in_channels)
        x = torch.randn(batch_size, seq_len, in_channels)
        output = model(x)
        
        assert output.shape[0] == batch_size
        assert len(output.shape) == 2  # (batch_size, embedding_dim)

    def test_attention_output(self) -> None:
        """Test attention output."""
        batch_size = 4
        seq_len = 24
        in_channels = 128
        
        model = LTAE(in_channels=in_channels, return_att=True)
        x = torch.randn(batch_size, seq_len, in_channels)
        output, attention = model(x)
        
        assert output.shape[0] == batch_size
        assert attention.shape[1] == batch_size  # (n_head, batch_size, seq_len)
        assert attention.shape[2] == seq_len

    @pytest.mark.parametrize("in_channels", [64, 128, 256])
    def test_different_channels(self, in_channels: int) -> None:
        """Test different input channel configurations."""
        batch_size = 4
        seq_len = 24
        
        model = LTAE(in_channels=in_channels)
        x = torch.randn(batch_size, seq_len, in_channels)
        output = model(x)
        
        assert output.shape[0] == batch_size 