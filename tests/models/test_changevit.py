# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.
"""ChangeViT model tests."""

import pytest
import torch

from torchgeo.models import ChangeViT
from torchgeo.models.changevit import (
    ChangeViTDecoder,
    DetailCaptureModule,
    FeatureInjector,
)

BATCH_SIZE = [1, 2]


class TestChangeViT:
    @torch.no_grad()
    @pytest.mark.parametrize('b', BATCH_SIZE)
    def test_forward(self, b: int) -> None:
        """Test ChangeViT forward pass with different batch sizes."""
        # Use tiny model with small images for fast testing
        model = ChangeViT(
            backbone='vit_tiny_patch16_224', img_size=64, pretrained=False
        )
        model.eval()

        # Input: [B, T=2, C=3, H, W]
        x = torch.randn(b, 2, 3, 64, 64)
        y = model(x)

        # Output: [B, 1, H, W] - binary change detection logits
        assert y.shape == (b, 1, 64, 64)

    @torch.no_grad()
    def test_different_img_sizes(self) -> None:
        """Test ChangeViT with different image sizes."""
        for img_size in [64, 128, 224]:
            model = ChangeViT(
                backbone='vit_tiny_patch16_224', img_size=img_size, pretrained=False
            )
            x = torch.randn(1, 2, 3, img_size, img_size)
            y = model(x)
            assert y.shape == (1, 1, img_size, img_size)

    @torch.no_grad()
    def test_components(self) -> None:
        """Test ChangeViT has required components."""
        model = ChangeViT(
            backbone='vit_tiny_patch16_224', img_size=64, pretrained=False
        )

        assert hasattr(model, 'encoder')
        assert hasattr(model, 'detail_capture')
        assert hasattr(model, 'feature_injector')
        assert hasattr(model, 'decoder')


class TestDetailCaptureModule:
    @torch.no_grad()
    def test_detail_capture_multiscale_output(self) -> None:
        """Test DetailCaptureModule returns 3 scales with correct channels."""
        dcm = DetailCaptureModule(in_channels=6)
        x = torch.randn(2, 6, 256, 256)

        c2, c3, c4 = dcm(x)

        # Check channel dimensions (paper specification: 64, 128, 256)
        assert c2.shape[1] == 64
        assert c3.shape[1] == 128
        assert c4.shape[1] == 256

        # Check spatial dimensions (1/2, 1/4, 1/8 of input)
        assert c2.shape[-2:] == (128, 128)  # 256 / 2
        assert c3.shape[-2:] == (64, 64)  # 256 / 4
        assert c4.shape[-2:] == (32, 32)  # 256 / 8


class TestFeatureInjector:
    @torch.no_grad()
    def test_feature_injector_output_shape(self) -> None:
        """Test FeatureInjector preserves ViT feature shape."""
        vit_dim = 384
        injector = FeatureInjector(vit_dim=vit_dim)

        # ViT features: [B, N_patches, D]
        vit_feats = torch.randn(2, 256, vit_dim)

        # Detail features at 3 scales
        detail_feats = (
            torch.randn(2, 64, 16, 16),  # 1/2 scale
            torch.randn(2, 128, 16, 16),  # 1/4 scale
            torch.randn(2, 256, 16, 16),  # 1/8 scale
        )

        enhanced_feats = injector(vit_feats, detail_feats)

        # Output should have same shape as input ViT features
        assert enhanced_feats.shape == vit_feats.shape


class TestChangeViTDecoder:
    @torch.no_grad()
    def test_decoder_output_shape(self) -> None:
        """Test ChangeViTDecoder returns bidirectional predictions."""
        decoder = ChangeViTDecoder(in_channels=384, num_classes=1)

        # Bitemporal features: [B, T=2, C, H, W]
        bi_features = torch.randn(2, 2, 384, 16, 16)

        c12, c21 = decoder(bi_features)

        # Both outputs should have same shape
        assert c12.shape == c21.shape
        assert c12.shape == (2, 1, 16, 16)
