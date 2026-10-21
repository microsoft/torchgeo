# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for the BidirectionalConvLSTM model."""

import pytest
import torch

from torchgeo.models import BidirectionalConvLSTM


class TestBidirectionalConvLSTM:
    """Tests for the BidirectionalConvLSTM model."""

    @pytest.mark.parametrize('lengths', [None, torch.tensor([4, 2])])
    def test_forward(self, lengths: torch.Tensor | None) -> None:
        """Test prediction and feature shapes."""
        x = torch.rand(2, 4, 3, 16, 16)
        model = BidirectionalConvLSTM(
            input_dim=3,
            hidden_dim=[8, 16],
            kernel_size=[3, (3, 3)],
            num_layers=2,
            num_classes=5,
        )

        features = model.forward_features(x, lengths=lengths)
        output = model(x, lengths=lengths)

        assert features.shape == (2, 32, 16, 16)
        assert output.shape == (2, 5, 16, 16)

    def test_padding_does_not_affect_output(self) -> None:
        """Test that values after each valid sequence prefix are ignored."""
        x = torch.rand(2, 4, 3, 8, 8)
        changed_padding = x.clone()
        changed_padding[1, 2:] = torch.rand_like(changed_padding[1, 2:]) + 10
        lengths = torch.tensor([4, 2])
        model = BidirectionalConvLSTM(input_dim=3, hidden_dim=4)

        actual = model(x, lengths=lengths)
        expected = model(changed_padding, lengths=lengths)

        torch.testing.assert_close(actual, expected)

    @pytest.mark.parametrize(
        'x, lengths, match',
        [
            (torch.rand(2, 3, 8, 8), None, 'Expected input_tensor'),
            (torch.rand(2, 4, 3, 8, 8), torch.tensor([4]), 'shape'),
        ],
    )
    def test_invalid_input(
        self, x: torch.Tensor, lengths: torch.Tensor | None, match: str
    ) -> None:
        """Test invalid input and sequence-length shapes."""
        model = BidirectionalConvLSTM(input_dim=3, hidden_dim=4)

        with pytest.raises(ValueError, match=match):
            model(x, lengths=lengths)
