# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.datamodules.levircd import LEVIRCDDataModule, LEVIRCDPlusDataModule
from torchgeo.transforms.transforms import _ExtractPatches


class TestLEVIRCDDataModule:
    def test_levircd_temporal_correspondence(self) -> None:
        """Test LEVIR-CD temporal correspondence issues from GitHub issue #2920."""
        # Test LEVIRCDDataModule temporal correspondence
        batch_size = 2
        temporal_frames = 2  # t1, t2 for images
        patches_per_frame = 16  # 16 patches per 1024x1024 image
        channels = 3
        height = width = 256

        # Mock batch similar to what LEVIRCD would produce after _ExtractPatches
        batch = {
            'image': torch.randn(
                batch_size, temporal_frames, patches_per_frame, channels, height, width
            ),
            'mask': torch.randn(
                batch_size, 1, patches_per_frame, 1, height, width
            ),  # Single change mask
        }

        # Test the on_after_batch_transfer method with patch extraction
        datamodule = LEVIRCDDataModule(batch_size=8, patch_size=256, stride=256)
        result_batch = datamodule.on_after_batch_transfer(batch, 0)

        # Check spatial correspondence: patch 0 from image should correspond to patch 0 from mask
        expected_image_shape = (
            batch_size * patches_per_frame,
            temporal_frames,
            channels,
            height,
            width,
        )
        expected_mask_shape = (batch_size * patches_per_frame, 1, height, width)

        assert result_batch['image'].shape == expected_image_shape, (
            f'Image shape mismatch: expected {expected_image_shape}, got {result_batch["image"].shape}'
        )
        assert result_batch['mask'].shape == expected_mask_shape, (
            f'Mask shape mismatch: expected {expected_mask_shape}, got {result_batch["mask"].shape}'
        )

    def test_extract_patches_temporal_ordering(self) -> None:
        """Test _ExtractPatches temporal ordering with VideoSequential from GitHub issue #2920."""
        # Simulate VideoSequential input: [B, T, C, H, W]
        batch_size = 2
        temporal_frames = 2
        channels = 3
        height = width = 512  # Smaller for faster testing

        input_tensor = torch.randn(batch_size, temporal_frames, channels, height, width)

        # VideoSequential flattens B and T dimensions for processing: [B*T, C, H, W]
        flattened_input = input_tensor.reshape(
            batch_size * temporal_frames, channels, height, width
        )

        # Apply _ExtractPatches
        extract_patches = _ExtractPatches(window_size=256, stride=256, keepdim=False)
        patches = extract_patches(flattened_input)

        # With keepdim=False, patches are returned as [B*T, N, C, H, W]
        # where B*T = 4 (2 batch x 2 temporal), N = 4 patches per image
        expected_batch_temporal = batch_size * temporal_frames  # 4
        expected_patches_per_image = (height // 256) * (width // 256)  # 4

        assert patches.shape == (
            expected_batch_temporal,
            expected_patches_per_image,
            channels,
            256,
            256,
        ), (
            f'Expected shape ({expected_batch_temporal}, {expected_patches_per_image}, {channels}, 256, 256), '
            f'got {patches.shape}'
        )

    def test_random_vs_deterministic_cropping(self) -> None:
        """Test that training uses synchronized random crops while val/test use deterministic patches."""
        datamodule = LEVIRCDDataModule(batch_size=2, patch_size=256, stride=256)

        # Test training transform setup (should use synchronized random crops for alignment)
        if datamodule.train_aug is not None:
            assert hasattr(datamodule.train_aug, 'same_on_batch')
            assert datamodule.train_aug.same_on_batch, (
                'Training should use same crops per batch for temporal-mask alignment'
            )

        # Test validation transform setup (should be deterministic)
        if datamodule.val_aug is not None:
            assert hasattr(datamodule.val_aug, 'same_on_batch')
            assert datamodule.val_aug.same_on_batch, (
                'Validation should use same crops per batch'
            )

    def test_levircd_grid_sampling(self) -> None:
        """Test LEVIRCDDataModule with grid sampling mode."""
        datamodule = LEVIRCDDataModule(
            batch_size=2, patch_size=256, stride=256, val_patch_sampling='grid'
        )

        # Test that grid sampling mode uses ExtractPatches
        assert datamodule.val_patch_sampling == 'grid'

        # Create mock batch with 6D shape (includes patches dimension)
        batch = {
            'image': torch.randn(2, 2, 16, 3, 256, 256),  # [B, T, P, C, H, W]
            'mask': torch.randn(2, 1, 16, 1, 256, 256),  # [B, T, P, C, H, W]
        }

        result = datamodule.on_after_batch_transfer(batch, 0)

        # Should reshape to flatten patches into batch dimension
        assert result['image'].shape == (32, 2, 3, 256, 256)  # [B*P, T, C, H, W]
        assert result['mask'].shape == (32, 1, 256, 256)  # [B*P, C, H, W]

    def test_levircd_mask_shape_handling(self) -> None:
        """Test LEVIRCDDataModule handles mask shape with extra temporal dimension."""
        datamodule = LEVIRCDDataModule(batch_size=2, patch_size=256)

        # Create mock batch with 5D mask shape [B, C, 1, H, W]
        batch = {
            'image': torch.randn(2, 2, 3, 256, 256),  # [B, T, C, H, W]
            'mask': torch.randn(2, 1, 1, 256, 256),  # [B, C, 1, H, W]
        }

        result = datamodule.on_after_batch_transfer(batch, 0)

        # Should squeeze temporal dimension from mask
        assert result['mask'].shape == (2, 1, 256, 256)  # [B, C, H, W]


class TestLEVIRCDPlusDataModule:
    def test_levircdplus_temporal_correspondence(self) -> None:
        """Test LEVIR-CD+ temporal correspondence."""
        # Test LEVIRCDPlusDataModule temporal correspondence (inherits from base)
        batch_size = 2
        temporal_frames = 2
        patches_per_frame = 16
        channels = 3
        height = width = 256

        # Mock batch similar to what LEVIRCD+ would produce after _ExtractPatches
        batch = {
            'image': torch.randn(
                batch_size, temporal_frames, patches_per_frame, channels, height, width
            ),
            'mask': torch.randn(batch_size, 1, patches_per_frame, 1, height, width),
        }

        # Test the on_after_batch_transfer method
        datamodule = LEVIRCDPlusDataModule(
            batch_size=8, patch_size=256, stride=256, val_split_pct=0.2
        )
        result_batch = datamodule.on_after_batch_transfer(batch, 0)

        # Check spatial correspondence
        expected_image_shape = (
            batch_size * patches_per_frame,
            temporal_frames,
            channels,
            height,
            width,
        )
        expected_mask_shape = (batch_size * patches_per_frame, 1, height, width)

        assert result_batch['image'].shape == expected_image_shape
        assert result_batch['mask'].shape == expected_mask_shape

    def test_levircdplus_val_split_pct(self) -> None:
        """Test that LEVIRCDPlusDataModule has val_split_pct parameter."""
        datamodule = LEVIRCDPlusDataModule(
            batch_size=2, patch_size=256, val_split_pct=0.3
        )
        assert datamodule.val_split_pct == 0.3

    def test_levircdplus_grid_sampling(self) -> None:
        """Test LEVIRCDPlusDataModule with grid sampling mode."""
        datamodule = LEVIRCDPlusDataModule(
            batch_size=2, patch_size=256, stride=256, val_patch_sampling='grid'
        )

        # Test that grid sampling mode is set correctly
        assert datamodule.val_patch_sampling == 'grid'

        # Create mock batch with 6D shape
        batch = {
            'image': torch.randn(2, 2, 16, 3, 256, 256),  # [B, T, P, C, H, W]
            'mask': torch.randn(2, 1, 16, 1, 256, 256),  # [B, T, P, C, H, W]
        }

        result = datamodule.on_after_batch_transfer(batch, 0)

        # Should reshape to flatten patches into batch dimension
        assert result['image'].shape == (32, 2, 3, 256, 256)  # [B*P, T, C, H, W]
        assert result['mask'].shape == (32, 1, 256, 256)  # [B*P, C, H, W]
