# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo temporal transforms."""

from typing import Any, Literal

import kornia.augmentation as K
from einops import rearrange
from torch import Tensor


class TemporalRearrange(K.IntensityAugmentationBase2D):
    """Rearrange temporal and channel dimensions.

    This transform allows conversion between:
    - B x T x C x H x W (temporal-explicit)
    - B x (T*C) x H x W (temporal-channel)
    """

    def __init__(
        self,
        mode: Literal['merge', 'split'],
        num_temporal_channels: int,
        p: float = 1.0,
        p_batch: float = 1.0,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        """Initialize a new TemporalRearrange instance.

        Args:
            mode: Whether to 'merge' (B x T x C x H x W -> B x TC x H x W) or
                'split' (B x TC x H x W -> B x T x C x H x W) temporal dimensions
            num_temporal_channels: Number of temporal channels (T) in the sequence
            p: Probability for applying the transform element-wise
            p_batch: Probability for applying the transform batch-wise
            same_on_batch: Apply the same transformation across the batch
            keepdim: Whether to keep the output shape the same as input
        """
        super().__init__(
            p=p, p_batch=p_batch, same_on_batch=same_on_batch, keepdim=keepdim
        )
        if mode not in ['merge', 'split']:
            raise ValueError("mode must be either 'merge' or 'split'")

        self.flags = {'mode': mode, 'num_temporal_channels': num_temporal_channels}

    def apply_transform(self, input: Tensor, flags: dict[str, Any]) -> Tensor:
        """Apply the transform.

        Args:
            input: Input tensor
            flags: Static parameters including mode and number of temporal channels

        Returns:
            Transformed tensor with rearranged dimensions

        Raises:
            ValueError: If input tensor dimensions don't match expected shape
        """
        mode = flags['mode']
        t = flags['num_temporal_channels']

        if mode == 'merge':
            if input.ndim != 5:
                raise ValueError(
                    f'Expected 5D input tensor (B,T,C,H,W), got shape {input.shape}'
                )
            return rearrange(input, 'b t c h w -> b (t c) h w')
        else:
            if input.ndim != 4:
                raise ValueError(
                    f'Expected 4D input tensor (B,TC,H,W), got shape {input.shape}'
                )
            tc = input.shape[1]
            if tc % t != 0:
                raise ValueError(
                    f'Input channels ({tc}) must be divisible by '
                    f'num_temporal_channels ({t})'
                )
            c = tc // t
            return rearrange(input, 'b (t c) h w -> b t c h w', t=t, c=c)
