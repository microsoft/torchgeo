# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo intensity transforms."""

import torch
from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor


class ToDecibelScale(IntensityAugmentationBase2D):
    """Convert input tensor to decibel scale.

    Primarily used for converting SAR pixel values to decibel scale.

    .. versionadded:: 0.8
    """

    def __init__(
        self, shift: float = 0.0, scale: float = 1.0, keepdim: bool = False
    ) -> None:
        """Initialize a new ToDecibelScale instance.

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


class Sentinel1ChangeMap(IntensityAugmentationBase2D):
    """Extracts a change map from Sentinel-1 SAR pre and post imagery.

    Adapted from https://github.com/microsoft/ai4g-flood. Copyright (c) 2024 Microsoft

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        vv_threshold: int = 100,
        vh_threshold: int = 90,
        vv_min_threshold: int = 75,
        vh_min_threshold: int = 70,
        delta_amplitude: float = 10,
        keepdim: bool = False,
    ) -> None:
        """Initializes a new Sentinel1ChangeMap instance.

        Args:
            vv_threshold: Threshold for VV band to detect change.
            vh_threshold: Threshold for VH band to detect change.
            vv_min_threshold: Minimum threshold for VV band to consider valid data.
            vh_min_threshold: Minimum threshold for VH band to consider valid data.
            delta_amplitude: Minimum change in amplitude to consider as a change.
            keepdim: Whether to keep the output shape the same as input (True)
                or broadcast it to the batch form (False).
        """
        super().__init__(p=1.0, p_batch=1.0, same_on_batch=True, keepdim=keepdim)
        self.flags = {
            'vv_threshold': vv_threshold,
            'vh_threshold': vh_threshold,
            'vv_min_threshold': vv_min_threshold,
            'vh_min_threshold': vh_min_threshold,
            'delta_amplitude': delta_amplitude,
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
            input: Input tensor of shape (N, 4, H, W) where N is the batch size,
                4 is the number of channels (VV pre, VH pre, VV post, VH post),
                and H, W are the height and width of the images.
            params: Generated parameters.
            flags: Static parameters.
            transform: The geometric transformation tensor.

        Returns:
            A tensor of shape (N, 2, H, W) containing the change maps for VV and VH bands.
            The values are 1 for change detected and 0 for no change.
        """
        vv_pre, vh_pre, vv_post, vh_post = (
            input[:, 0],
            input[:, 1],
            input[:, 2],
            input[:, 3],
        )
        vv_change = (
            (vv_post < self.flags['vv_threshold'])
            & (vv_pre > self.flags['vv_threshold'])
            & ((vv_pre - vv_post) > self.flags['delta_amplitude'])
        ).to(torch.int)
        vh_change = (
            (vh_post < self.flags['vh_threshold'])
            & (vh_pre > self.flags['vh_threshold'])
            & ((vh_pre - vh_post) > self.flags['delta_amplitude'])
        ).to(torch.int)

        zero_idx = (
            (vv_post < self.flags['vv_min_threshold'])
            | (vv_pre < self.flags['vv_min_threshold'])
            | (vh_post < self.flags['vh_min_threshold'])
            | (vh_pre < self.flags['vh_min_threshold'])
        )
        vv_change[zero_idx] = 0
        vh_change[zero_idx] = 0
        change = torch.stack([vv_change, vh_change], dim=1).to(torch.float)

        return change
