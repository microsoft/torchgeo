# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo temporal transforms."""

from typing import Any

from einops import rearrange
from kornia.augmentation._3d.geometric.base import GeometricAugmentationBase3D
from torch import Tensor


class Rearrange(GeometricAugmentationBase3D):
    """Rearrange tensor between [B, T, C, H, W] and [B, T*C, H, W]."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize a Rearrange instance.

        Args:
            *args: Positional arguments for einops.rearrange
            **kwargs: Keyword arguments for einops.rearrange
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
