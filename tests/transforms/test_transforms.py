# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import torch

from torchgeo.transforms.transforms import _ExtractPatches


def test_extract_patches() -> None:
    b, c, h, w = 2, 3, 64, 64
    p = 32
    s = p
    num_patches = ((h - p + s) // s) * ((w - p + s) // s)

    # test default settings (when stride is not defined, s=p)
    batch = {
        'image': torch.randn(size=(b, c, h, w)),
        'mask': torch.randint(low=0, high=2, size=(b, h, w)),
    }
    train_transforms = K.AugmentationSequential(
        _ExtractPatches(window_size=p), same_on_batch=True, data_keys=None
    )
    output = train_transforms(batch)
    assert output['image'].shape == (b * num_patches, c, p, p)
    assert output['mask'].shape == (b * num_patches, 1, p, p)

    # Test different stride
    s = 16
    num_patches = ((h - p + s) // s) * ((w - p + s) // s)
    batch = {
        'image': torch.randn(size=(b, c, h, w)),
        'mask': torch.randint(low=0, high=2, size=(b, h, w)),
    }
    train_transforms = K.AugmentationSequential(
        _ExtractPatches(window_size=p, stride=s), same_on_batch=True, data_keys=None
    )
    output = train_transforms(batch)
    assert output['image'].shape == (b * num_patches, c, p, p)
    assert output['mask'].shape == (b * num_patches, 1, p, p)

    # Test keepdim=False
    s = p
    num_patches = ((h - p + s) // s) * ((w - p + s) // s)
    batch = {
        'image': torch.randn(size=(b, c, h, w)),
        'mask': torch.randint(low=0, high=2, size=(b, h, w)),
    }
    train_transforms = K.AugmentationSequential(
        _ExtractPatches(window_size=p, stride=s, keepdim=False),
        same_on_batch=True,
        data_keys=None,
    )
    output = train_transforms(batch)
    for k, v in output.items():
        print(k, v.shape, v.dtype)
    assert output['image'].shape == (b, num_patches, c, p, p)
    assert output['mask'].shape == (b, num_patches, 1, p, p)


def test_extract_patches_temporal_ordering() -> None:
    """Test _ExtractPatches temporal ordering with VideoSequential from GitHub issue #2920."""
    # Simulate VideoSequential input: [B, T, C, H, W]
    batch_size = 2
    temporal_frames = 2
    channels = 3
    height = width = 512

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
