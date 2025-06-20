# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo intensity transforms."""

from collections.abc import Sequence

import torch
from einops import rearrange
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor


class PowerToDecibel(IntensityAugmentationBase2D):
    """Convert input tensor of power values to decibel scale.

    Primarily used for converting SAR pixel values to decibel scale.

    .. versionadded:: 0.8
    """

    def __init__(
        self, shift: float = 0.0, scale: float = 1.0, keepdim: bool = False
    ) -> None:
        """Initialize a new PowerToDecibel instance.

        Args:
            shift: The value to add to the decibel scale.
                This is useful for avoiding negative values.
            scale: An additional scale factor to apply to the decibel value.
            keepdim: Whether to keep the output shape the same as input (True)
                or broadcast it to the batch form (False).
        """
        super().__init__(p=1.0, p_batch=1.0, same_on_batch=True, keepdim=keepdim)
        self.flags = {'scale': scale, 'shift': shift}

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Tensor],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: The input tensor.
            params: Generated parameters.
            flags: Static parameters.
            transform: The geometric transformation tensor.

        Returns:
            The transformed input.
        """
        out = flags['scale'] * 10 * torch.log10(input + 1e-6) + flags['shift']
        return out


class ToThresholdedChangeMask(IntensityAugmentationBase2D):
    """Extracts a change mask from pre and post imagery.

    This transform computes the difference between 2 stacked images along the channel
    dimension and return a pixelwise boolean change map indicating 'change'
    or 'no change' based on difference values greater than a change threshold.

    Adapted from https://github.com/microsoft/ai4g-flood. Copyright (c) 2024 Microsoft

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        change_thresholds: Sequence[float],  # [10, 10]
        thresholds: Sequence[float] | None = None,  # [100, 90]
        min_thresholds: Sequence[float] | None = None,  # [75, 70]
        keepdim: bool = False,
    ) -> None:
        """Initializes a new ToChangeMask instance.

        Args:
            change_thresholds: Thresholds determine pixelwise change.
            thresholds: Thresholds for pre and post image channels.
            min_thresholds: Minimum thresholds for pre and post image channels.
            keepdim: Whether to keep the output shape the same as input (True)
                or broadcast it to the batch form (False).
        """
        super().__init__(p=1.0, p_batch=1.0, same_on_batch=True, keepdim=keepdim)
        self.flags = {
            'change_thresholds': torch.tensor(change_thresholds),
            'thresholds': torch.tensor(thresholds) if thresholds else None,
            'min_thresholds': torch.tensor(min_thresholds) if min_thresholds else None,
        }

    def apply_transform(
        self,
        input: Tensor,
        params: dict[str, Tensor],
        flags: dict[str, Tensor],
        transform: Tensor | None = None,
    ) -> Tensor:
        """Apply the transform.

        Args:
            input: Input tensor of shape (N, 2*C, H, W) where N is the batch size,
                2*C is the number of channels (pre and post images), and H, W are
                the height and width of the images.
            params: Generated parameters.
            flags: Static parameters.
            transform: The geometric transformation tensor.

        Returns:
            A tensor of shape (N, C, H, W) containing the pixelwise binary change masks.
            The values are 1 for change detected and 0 for no change.
        """
        b, c, h, w = input.shape
        x = rearrange(input, 'b (t c) h w -> b t c h w', t=2, c=c // 2)
        pre, post = x[:, 0], x[:, 1]

        change_mask = torch.abs(post - pre) > self.flags['change_thresholds']

        if self.flags['thresholds'] is not None:
            pre_mask = pre > self.flags['thresholds']
            post_mask = post > self.flags['thresholds']
            change_mask = change_mask & pre_mask & post_mask

        change_mask = change_mask.to(torch.int)

        if self.flags['min_thresholds'] is not None:
            mask = (
                pre < self.flags['min_thresholds'] | post < self.flags['min_thresholds']
            )
            change_mask[mask] = 0

        return change_mask.to(torch.float)
