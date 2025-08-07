# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo transforms."""

from typing import Any

import kornia.augmentation as K
import torch
from einops import rearrange
from kornia.contrib import extract_tensor_patches
from torch import Tensor
from typing_extensions import deprecated


@deprecated('Use kornia.augmentation.AugmentationSequential instead')
class AugmentationSequential(K.AugmentationSequential):
    """Deprecated wrapper around kornia.augmentation.AugmentationSequential."""


# TODO: contribute these to Kornia and delete this file


class _ExtractPatches(K.GeometricAugmentationBase2D):
    """Extract patches from an image or mask."""

    def __init__(
        self,
        window_size: int | tuple[int, int],
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] | None = 0,
        keepdim: bool = True,
    ) -> None:
        """Initialize a new _ExtractPatches instance.

        Args:
            window_size: desired output size (out_h, out_w) of the crop
            stride: stride of window to extract patches. Defaults to non-overlapping
                patches (stride=window_size)
            padding: zero padding added to the height and width dimensions
            keepdim: Combine the patch dimension into the batch dimension
        """
        super().__init__(p=1)
        self.flags = {
            'window_size': window_size,
            'stride': stride if stride is not None else window_size,
            'padding': padding,
            'keepdim': keepdim,
        }

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
        out: Tensor = self.identity_matrix(input)
        return out

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        out = extract_tensor_patches(
            input,
            window_size=flags['window_size'],
            stride=flags['stride'],
            padding=flags['padding'],
        )

        # Handle temporal data from VideoSequential
        # If we extracted multiple patches and batch size suggests temporal data (even number),
        # rearrange to group patches by spatial location rather than temporal sequence
        # Only apply this fix when keepdim=True (for flattening compatibility)
        if (
            len(out.shape) == 5
            and out.shape[1] > 1
            and input.shape[0] % 2 == 0
            and flags['keepdim']
        ):
            # Fix Issue 3: Rearrange to group patches by spatial location
            # Current: patches from t1, then patches from t2
            # Desired: patch_0 from [t1,t2], patch_1 from [t1,t2], etc.
            temporal_frames = 2
            out = rearrange(out, '(b t) n c h w -> (b n t) c h w', t=temporal_frames)
        elif flags['keepdim']:
            # Original behavior - flatten patches into batch dimension
            out = rearrange(out, 'b n c h w -> (b n) c h w')
        # If keepdim=False, keep the [B, N, C, H, W] shape as is

        return out


class _Clamp(K.IntensityAugmentationBase2D):
    """Clamp images to a specific range."""

    def __init__(
        self,
        p: float = 0.5,
        p_batch: float = 1,
        min: float = 0,
        max: float = 1,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        """Initialize a new _Clamp instance.

        Args:
            p: Probability for applying an augmentation. This param controls the
                augmentation probabilities element-wise for a batch.
            p_batch: Probability for applying an augmentation to a batch. This param
                controls the augmentation probabilities batch-wise.
            min: Minimum value to clamp to.
            max: Maximum value to clamp to.
            same_on_batch: Apply the same transformation across the batch.
            keepdim: Whether to keep the output shape the same as input ``True``
                or broadcast it to the batch form ``False``.
        """
        super().__init__(
            p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
        )
        self.flags = {'min': min, 'max': max}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Any],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: the input tensor
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        return torch.clamp(input, self.flags['min'], self.flags['max'])
