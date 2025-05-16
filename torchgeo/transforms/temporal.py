# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo temporal transforms."""

from einops import rearrange
from torch import Tensor


class TemporalToChannels:
    """Reshape tensor from [B, T, C, H, W] to [B, T*C, H, W]."""

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the transform.

        Args:
            x (Tensor): Input tensor of shape [B, T, C, H, W].

        Returns:
            Tensor: Output tensor of shape [B, T*C, H, W], where the temporal (T)
                and channel (C) dimensions are merged into a single channel dimension.
        """
        if x.ndim != 5:
            raise ValueError(f'Expected 5D tensor [B, T, C, H, W], got {x.shape}')
        return rearrange(x, 'b t c h w -> b (t c) h w')


class ChannelsToTemporal:
    """Reshape tensor from [B, T*C, H, W] to [B, T, C, H, W]."""

    def __init__(self, T: int, C: int) -> None:
        """Initialize the ChannelsToTemporal transform.

        Args:
            T (int): Number of temporal steps (time dimension) in the output tensor.
            C (int): Number of channels per temporal step in the output tensor.
        """
        self.T = T
        self.C = C

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the transform.

        Args:
            x (Tensor): Input tensor of shape [B, T*C, H, W].

        Returns:
            Tensor: Output tensor of shape [B, T, C, H, W], where the channel dimension
                is split into temporal (T) and channel (C) dimensions.
        """
        if x.ndim != 4:
            raise ValueError(f'Expected 4D tensor [B, T*C, H, W], got {x.shape}')
        B, TC, H, W = x.shape
        if TC != self.T * self.C:
            raise ValueError(
                f'Channel dimension ({TC}) does not match T*C ({self.T}*{self.C}={self.T * self.C})'
            )
        return rearrange(x, 'b (t c) h w -> b t c h w', t=self.T, c=self.C)
