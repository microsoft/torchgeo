# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.models import TSViT


class TestTSViT:
    @pytest.fixture
    def model(self) -> TSViT:
        return TSViT(
            img_res=24,
            patch_size=3,
            num_channels=14,
            num_classes=20,
            max_seq_len=16,
            dim=64,  # Smaller dim for faster tests
            temporal_depth=2,  # Shallow depth for faster tests
            spatial_depth=2,
        )

    def test_forward(self, model: TSViT) -> None:
        # Create dummy batch of Satellite Image Time Series
        # Shape: (Batch, Time, Height, Width, Channels)
        x = torch.rand(2, 16, 24, 24, 14)
        y = model(x)

        # Expected output shape: (Batch, Classes, Height, Width)
        assert y.shape == (2, 20, 24, 24)
