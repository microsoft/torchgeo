# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo temporal transforms."""

import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from torch import Tensor


class Rearrange(GeometricAugmentationBase3D):
    """Rearrange tensor dimensions.

    Examples:
        To insert a time dimension::

            Rearrange('b (t c) h w -> b t c h w', c=1)

        To collapse the time dimension::

            Rearrange('b t c h w -> b (t c) h w')
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a Rearrange instance.

        Args:
            *args: Positional arguments for :func:`einops.rearrange`.
            **kwargs: Keyword arguments for :func:`einops.rearrange`.
        """
        super().__init__(p=1)
        self.flags = {'args': args, 'kwargs': kwargs}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the rearrangement to the input tensor.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            The rearranged tensor.
        """
        return rearrange(input, *flags['args'], **flags['kwargs'])

    def compute_transformation(
        self, input: Tensor, params: dict[str, Tensor], flags: dict[str, Any]
    ) -> Tensor:
        """Compute the transformation.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters

        Returns:
            the transformation
        """
        out = self.identity_matrix(input)
        return out


class TemporalEmbedding(nn.Module):
    """Generic sinusoidal embedding for periodic temporal features."""

    def __init__(self, period: int) -> None:
        """Initialize a TemporalEmbedding instance.

        Args:
            period: The period of the sinusoidal function.
        """
        super().__init__()
        self.period = period

    def forward(self, t: Tensor) -> Tensor:
        """Args:
            t: Tensor of shape (B,) or (B, 1), representing time values.

        Returns:
            Tensor of shape (B, 2), sin and cos embeddings.
        """
        t = t.view(-1, 1).float()
        scaled = 2 * math.pi * t / self.period
        return torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
