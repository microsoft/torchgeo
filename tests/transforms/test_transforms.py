# Copyright (c) Microsoft Corporation. All rights reserved.
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
