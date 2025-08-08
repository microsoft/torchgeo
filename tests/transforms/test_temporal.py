# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import kornia.augmentation as K
import torch

from torchgeo.transforms.temporal import CyclicalEncoder, Rearrange


def test_rearrange_combine() -> None:
    """Test rearrange: [B, T, C, H, W] -> [B, T*C, H, W]."""
    b, t, c, h, w = 2, 4, 3, 16, 16
    x = torch.randn(size=(b, t, c, h, w))
    rearranger = Rearrange('b t c h w -> b (t c) h w')
    out = rearranger(x)
    assert out.shape == (b, t * c, h, w), (
        f'Expected {(b, t * c, h, w)}, got {out.shape}'
    )


def test_rearrange_split() -> None:
    """Test rearrange: [B, T*C, H, W] -> [B, T, C, H, W]."""
    b, t, c_original, h, w = 2, 4, 3, 16, 16
    x_combined = torch.randn((b, t * c_original, h, w))
    forward_pattern = 'b (t c) h w -> b t c h w'
    rearranger = Rearrange(forward_pattern, c=c_original)
    out = rearranger.apply_transform(
        x_combined, params={}, flags=rearranger.flags, transform=None
    )
    assert out.shape == (b, t, c_original, h, w)


def test_rearrange_integration_in_augmentation_sequential() -> None:
    """Test Rearrange integration within Kornia's AugmentationSequential."""
    b, t, c, h, w = 2, 4, 3, 16, 16
    x = torch.randn(size=(b, t, c, h, w))
    batch = {'image': x}
    train_transforms = K.AugmentationSequential(
        Rearrange('b t c h w -> b (t c) h w'), same_on_batch=True, data_keys=None
    )
    out_batch = train_transforms(batch)
    assert 'image' in out_batch
    assert out_batch['image'].shape == (b, t * c, h, w)


def test_temporal_embedding_shape_and_values() -> None:
    period = 365
    model = CyclicalEncoder(period=period)
    t = torch.tensor([0, 91, 182, 273, 364])
    output = model(t)
    norms = torch.norm(output, dim=-1)
    assert output.shape == (5, 2), f'Expected shape (5, 2), got {output.shape}'
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)
    assert torch.allclose(output[0], torch.tensor([1.0, 0.0]), atol=1e-4)

def test_temporal_embedding_batch_dimension() -> None:
    model = CyclicalEncoder(period=24)
    t = torch.arange(0, 24).unsqueeze(1)
    output = model(t)
    assert output.shape == (24, 2)
