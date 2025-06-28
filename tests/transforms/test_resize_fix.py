"""Test for K.Resize fix to ensure square resizing behavior."""

import torch
import kornia.augmentation as K
import pytest


class TestResizeFix:
    """Test class to validate K.Resize fix for square dimensions."""

    def test_resize_square_behavior(self) -> None:
        """Test that K.Resize((X, X)) produces square outputs."""
        # Test with non-square input
        x = torch.randn(1, 3, 128, 320)
        
        # Test correct behavior with tuple argument
        transform_correct = K.Resize((256, 256))
        result_correct = transform_correct(x)
        
        # Should produce exactly 256x256
        assert result_correct.shape == (1, 3, 256, 256)
        
    def test_resize_with_different_aspect_ratios(self) -> None:
        """Test resize behavior with various aspect ratios."""
        test_cases = [
            (100, 400),  # 1:4 aspect ratio
            (400, 100),  # 4:1 aspect ratio 
            (200, 200),  # 1:1 aspect ratio
            (150, 300),  # 1:2 aspect ratio
        ]
        
        target_size = 224
        transform = K.Resize((target_size, target_size))
        
        for h, w in test_cases:
            x = torch.randn(1, 3, h, w)
            result = transform(x)
            assert result.shape == (1, 3, target_size, target_size), \
                f"Failed for input {h}x{w}, got {result.shape}"

    def test_model_transforms_produce_expected_shapes(self) -> None:
        """Test that model transforms produce expected square outputs."""
        from torchgeo.models.resnet import (
            _ssl4eo_s12_transforms_s1,
            _ssl4eo_s12_transforms_s2_10k,
            _ssl4eo_s12_transforms_s2_stats,
            _seco_transforms,
            _gassl_transforms,
        )
        
        # Test S1 transforms (2 channels)
        x_s1 = torch.randn(1, 2, 128, 320)
        result_s1 = _ssl4eo_s12_transforms_s1({'image': x_s1})
        assert result_s1['image'].shape[-2:] == (224, 224)
        
        # Test S2 transforms (13 channels)
        x_s2 = torch.randn(1, 13, 128, 320)
        result_s2_10k = _ssl4eo_s12_transforms_s2_10k({'image': x_s2})
        assert result_s2_10k['image'].shape[-2:] == (224, 224)
        
        result_s2_stats = _ssl4eo_s12_transforms_s2_stats({'image': x_s2})
        assert result_s2_stats['image'].shape[-2:] == (224, 224)
        
        # Test SECO transforms (RGB)
        x_rgb = torch.randn(1, 3, 128, 320)
        result_seco = _seco_transforms({'image': x_rgb})
        assert result_seco['image'].shape[-2:] == (224, 224)
        
        # Test GASSL transforms (RGB) - this one only resizes to 224, no crop
        result_gassl = _gassl_transforms({'image': x_rgb})
        assert result_gassl['image'].shape[-2:] == (224, 224)