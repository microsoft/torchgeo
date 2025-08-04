# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo temporal transforms."""

import math
from typing import Any

import pandas as pd
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


class CyclicalEncoder(nn.Module):
    """Generic sinusoidal embedding for periodic temporal features.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        period: pd.Timedelta,
        time_key: str = 'time',
    ) -> None:
        """Initialize a CyclicalEncoder instance.

        Args:
            period: The period of the sinusoidal function.
            time_key: The key in the input data containing time values.
        """
        super().__init__()
        self.period = period
        self.time_key = time_key

    def forward(self, sample: dict[str, Any]) -> dict[str, Any]:
        """Add sinusoidal embeddings to the sample using the given time key."""
        t = sample[self.time_key]
        if isinstance(t, pd.Timestamp):
            t = t.timestamp()
        elif isinstance(t, int):
            t = float(t)
        else:
            raise TypeError(f'Unsupported type for time key {self.time_key}: {type(t)}')

        scaled = torch.tensor(
            2 * math.pi * t / self.period.total_seconds(), dtype=torch.float32
        ).unsqueeze(0)
        sample[f"sin_{self.time_key}"] = torch.sin(scaled)
        sample[f"cos_{self.time_key}"] = torch.cos(scaled)
        return sample
