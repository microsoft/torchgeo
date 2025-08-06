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
class _RandomNCrop(K.GeometricAugmentationBase2D):
    """Extract one random crop per image from many possible locations.

    Provides maximum training diversity by sampling different spatial locations
    across epochs while maintaining consistent batch sizes.
    """

    def __init__(self, size: int | tuple[int, int], pad_if_needed: bool = True) -> None:
        """Initialize a new _RandomNCrop instance.

        Args:
            size: desired output size (out_h, out_w) of the crop
            pad_if_needed: pad the image if crop size is larger than image size
        """
        super().__init__(p=1)
        self.flags = {
            'size': size if isinstance(size, tuple) else (size, size),
            'pad_if_needed': pad_if_needed,
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
        crop_h, crop_w = flags['size']

        # Handle different input shapes
        # When used inside VideoSequential, input comes as [B*T, C, H, W] (flattened)
        # When used directly on temporal data, input comes as [B, T, C, H, W]
        if len(input.shape) == 5:
            # Direct temporal data: [B, T, C, H, W]
            batch_size, temporal_frames, channels, height, width = input.shape
            # Flatten temporal dimension for processing: [B*T, C, H, W]
            input_flat = input.view(
                batch_size * temporal_frames, channels, height, width
            )
            is_temporal = True
        else:
            # Regular data or data from VideoSequential: [B, C, H, W] or [B*T, C, H, W]
            batch_size_flat, channels, height, width = input.shape
            input_flat = input
            is_temporal = False
            batch_size = batch_size_flat  # Could be B or B*T from VideoSequential

        # Pad if needed
        if flags['pad_if_needed']:
            pad_h = max(0, crop_h - height)
            pad_w = max(0, crop_w - width)
            if pad_h > 0 or pad_w > 0:
                input_flat = torch.nn.functional.pad(
                    input_flat, (0, pad_w, 0, pad_h), mode='constant', value=0
                )
                height, width = input_flat.shape[2], input_flat.shape[3]

        # Generate random crop positions
        max_y = height - crop_h
        max_x = width - crop_w

        if max_y < 0 or max_x < 0:
            raise ValueError(
                f'Crop size {flags["size"]} is larger than image size ({height}, {width}) even after padding'
            )

        # Generate random crop positions
        actual_batch_size = input_flat.shape[0]

        if is_temporal:
            # For direct temporal data, use the same crop position for all temporal frames of each sample
            base_batch_size = batch_size
            y_positions = torch.randint(
                0, max_y + 1, (base_batch_size,), device=input_flat.device
            )
            x_positions = torch.randint(
                0, max_x + 1, (base_batch_size,), device=input_flat.device
            )
            # Repeat positions for all temporal frames
            y_positions = y_positions.repeat_interleave(temporal_frames)
            x_positions = x_positions.repeat_interleave(temporal_frames)
        else:
            # For regular data or data from VideoSequential, each item gets its own crop
            # VideoSequential case: if B=2, T=2, we get B*T=4 items, each gets own crop
            y_positions = torch.randint(
                0, max_y + 1, (actual_batch_size,), device=input_flat.device
            )
            x_positions = torch.randint(
                0, max_x + 1, (actual_batch_size,), device=input_flat.device
            )

        # Extract crops
        crops = []
        for i in range(actual_batch_size):
            y_start = int(y_positions[i].item())
            x_start = int(x_positions[i].item())
            crop = input_flat[
                i : i + 1, :, y_start : y_start + crop_h, x_start : x_start + crop_w
            ]
            crops.append(crop)

        out = torch.cat(crops, dim=0)

        # Reshape back to temporal format if input was temporal
        if is_temporal:
            out = out.view(batch_size, temporal_frames, channels, crop_h, crop_w)

        return out


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
        if len(out.shape) == 5 and out.shape[1] > 1 and input.shape[0] % 2 == 0:
            # Fix Issue 3: Rearrange to group patches by spatial location
            # Current: patches from t1, then patches from t2
            # Desired: patch_0 from [t1,t2], patch_1 from [t1,t2], etc.
            temporal_frames = 2
            out = rearrange(out, '(b t) n c h w -> (b n t) c h w', t=temporal_frames)
        elif flags['keepdim']:
            # Original behavior - flatten patches into batch dimension
            out = rearrange(out, 'b n c h w -> (b n) c h w')

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
