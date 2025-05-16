# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.transforms.temporal import ChannelsToTemporal, TemporalToChannels

def test_temporal_to_channels():
    b, t, c, h, w = 2, 4, 3, 64, 64
    x = torch.randn(b, t, c, h, w)
    transform = TemporalToChannels()
    y = transform(x)
    assert y.shape == (b, t * c, h, w)
    x_recover = ChannelsToTemporal(T=t, C=c)(y)
    assert torch.allclose(x, x_recover, atol=1e-6)

def test_temporal_to_channels_wrong_dim():
    x = torch.randn(2, 3, 64, 64)
    transform = TemporalToChannels()
    try:
        _ = transform(x)
        assert False, "Expected ValueError for 4D input"
    except ValueError:
        pass

def test_channels_to_temporal():
    b, t, c, h, w = 2, 4, 3, 64, 64
    x = torch.randn(b, t * c, h, w)
    transform = ChannelsToTemporal(T=t, C=c)
    y = transform(x)
    assert y.shape == (b, t, c, h, w)
    x_recover = TemporalToChannels()(y)
    assert torch.allclose(x, x_recover, atol=1e-6)

def test_channels_to_temporal_wrong_dim():
    x = torch.randn(2, 4, 3, 64, 64)
    transform = ChannelsToTemporal(T=4, C=3)
    try:
        _ = transform(x)
        assert False, "Expected ValueError for 5D input"
    except ValueError:
        pass

def test_channels_to_temporal_wrong_size():
    x = torch.randn(2, 13, 64, 64)
    transform = ChannelsToTemporal(T=4, C=3)
    try:
        _ = transform(x)
        assert False, "Expected ValueError for channel mismatch"
    except ValueError:
        pass
