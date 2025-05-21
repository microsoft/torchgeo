# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo augmentations."""

from typing import Any

import kornia.augmentation as K
import torch
from einops import rearrange
from torch import Tensor


def sat_slidemix(input: Tensor, gamma: int, beta: float) -> Tensor:
    """Applies the Sat-SlideMix augmentation to a batch of images.

    Sat-SlideMix rolls (circularly shifts) images along either the height
    or width axis by a random amount.

    Args:
        input: Input batch of images with shape [B, C, H, W].
        gamma: The number of augmented samples to create for each
                     input image. The output batch size will be gamma * B.
        beta: The maximum percentage (0.0 to 1.0) of the image
                      dimension (height or width) to shift.

    Returns:
        Batch of augmented images with shape [gamma * B, C, H, W].
            This operation is label-preserving, so labels only need
            to be repeated gamma times by the caller.

    .. versionadded:: 0.8
    """
    if not isinstance(gamma, int) or gamma < 1:
        raise ValueError('gamma must be an integer >= 1')
    if not isinstance(beta, float) or not (0.0 <= beta <= 1.0):
        raise ValueError('beta must be a float between 0.0 and 1.0')
    if input.dim() != 4:
        raise ValueError('Input tensor must have 4 dimensions [B, C, H, W]')
    if beta == 0.0:  # No shift needed if beta is 0
        if gamma == 1:
            return input  # Return original if no augmentation needed
        else:
            # Just repeat the images if beta is 0 but gamma > 1
            return input.repeat_interleave(gamma, dim=0)

    B, _, H, W = input.shape
    device = input.device
    total_augmented_batch_size = B * gamma

    # Step 2: Repeat the input images gamma times
    # Using repeat_interleave is often clearer than repeat for this pattern
    # Shape becomes [B*gamma, C, H, W]
    augmented_imgs = input.repeat_interleave(gamma, dim=0)

    # Step 1: Sample magnitudes (as percentages) for each augmented image
    # The algorithm description implies uniform sampling from [0.0, beta]
    # torch.rand samples from [0, 1), so scale by beta
    magnitudes_percent = torch.rand(total_augmented_batch_size, device=device) * beta

    # Create tensors to store parameters for each image in the augmented batch
    pixel_shifts = torch.zeros(
        total_augmented_batch_size, dtype=torch.int64, device=device
    )
    shift_dims = torch.zeros(
        total_augmented_batch_size, dtype=torch.int64, device=device
    )  # 2 for H, 3 for W

    # Step 3 & 4 & 5 & 6: Loop through each image in the augmented batch
    # Vectorizing this is tricky because torch.roll needs specific shifts and dims
    # per element, which it doesn't directly support in a single call for varying
    # parameters across the batch dimension easily. A loop is straightforward.

    # Pre-calculate random choices to potentially use tensors later if possible
    # Dim choice: 0 for Height (dim=2), 1 for Width (dim=3)
    dim_choices = torch.randint(0, 2, (total_augmented_batch_size,), device=device)
    # Direction choice: -1 or 1
    directions = (
        torch.randint(0, 2, (total_augmented_batch_size,), device=device) * 2.0 - 1.0
    )

    # Calculate pixel shifts and target dimensions
    for i in range(total_augmented_batch_size):
        magnitude = magnitudes_percent[i]
        direction = directions[i]

        # Step 4: Randomly select dimension (H or W)
        if dim_choices[i] == 0:  # Shift Height
            dim_size = H
            shift_dims[i] = 2  # Dimension index in the 4D tensor [B, C, H, W]
        else:  # Shift Width
            dim_size = W
            shift_dims[i] = 3  # Dimension index in the 4D tensor [B, C, H, W]

        # Calculate shift amount in pixels
        # Ensure it's an integer for torch.roll
        shift_amount = torch.round(magnitude * dim_size * direction).int().item()
        pixel_shifts[i] = shift_amount

    # Step 6: Apply the rolls
    # It's necessary to roll each image individually if shifts/dims vary
    rolled_imgs = torch.empty_like(augmented_imgs)
    for i in range(total_augmented_batch_size):
        shift = int(pixel_shifts[i].item())
        dim_to_shift = int(shift_dims[i].item())  # 2 for H, 3 for W

        # Apply roll to the i-th image along the chosen dimension
        rolled_imgs[i] = torch.roll(
            augmented_imgs[i], shifts=shift, dims=dim_to_shift - 1
        )
        # Note: We roll augmented_imgs[i] which has shape [C, H, W].
        # In this 3D tensor, H is dim 1 and W is dim 2.
        # So we use dim_to_shift - 1 as the dimension index for torch.roll.

    # Step 9: Return the batch of rolled images
    return rolled_imgs


class SatSlideMix(K.GeometricAugmentationBase2D):
    """Extract patches from an image or mask."""

    def __init__(
        self, gamma: int = 1, beta: float = 0.3, keepdim: bool = True, p: float = 1.0
    ) -> None:
        """Initialize a new _ExtractPatches instance.

        Args:
            gamma: The number of augmented samples to create for each
                        input image. The output batch size will be gamma * B.
            beta: The maximum percentage (0.0 to 1.0) of the image
                        dimension (height or width) to shift.
            keepdim: Combine the patch dimension into the batch dimension
            p: Probability to apply the augmentation on each sample
        """
        super().__init__(p=p)
        self.flags = {'gamma': gamma, 'beta': beta, 'keepdim': keepdim}

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
        out = sat_slidemix(input, beta=flags['beta'], gamma=flags['gamma'])

        if flags['keepdim']:
            out = rearrange(out, 'b t c h w -> (b t) c h w')

        return out
