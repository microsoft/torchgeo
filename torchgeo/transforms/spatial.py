# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo augmentations."""

from typing import Any

import kornia.augmentation as K
import torch
from kornia.augmentation.random_generator import PlainUniformGenerator
from torch import Tensor


class SatSlideMix(K.GeometricAugmentationBase2D):
    """Applies the Sat-SlideMix augmentation to a batch of images and masks.

    Sat-SlideMix rolls (circularly shifts) images along either the height
    or width axis by a random amount.

    If you use this method in your research, please cite the following paper:

    * https://doi.org/10.1609/aaai.v39i27.35028

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        gamma: int = 1,
        beta: Tensor | float | tuple[float, float] | list[float] = (0.0, 1.0),
        p: float = 0.5,
    ) -> None:
        """Initialize a new SatSlideMix instance.

        Args:
            gamma: The number of augmented samples to create for each
                input image. The output batch size will be gamma * B.
            beta: The range of percentage (0.0 to 1.0) of the image
                dimension (height or width) to shift.
            p: Probability to apply the augmentation on each sample

        Raises:
            AssertionError: If `gamma` is not a positive integer.
        """
        super().__init__(p=p)
        assert isinstance(gamma, int) and gamma > 0, 'gamma must be a positive integer'
        self._param_generator: PlainUniformGenerator = PlainUniformGenerator(
            (beta, 'beta', 0.5, (0.0, 1.0)),
            ((0.0, 1.0), 'dim', 0.5, (0.0, 1.0)),
            ((0.0, 1.0), 'direction', 0.5, (0.0, 1.0)),
        )
        self.flags = {'gamma': gamma}

    def generate_parameters(self, batch_shape: tuple[int, ...]) -> dict[str, Tensor]:
        """Generate parameters for the batch."""
        B, C, H, W = batch_shape
        batch_shape = torch.Size((B * self.flags['gamma'], C, H, W))
        params: dict[str, Tensor] = self._param_generator(
            batch_shape, self.same_on_batch
        )
        return params

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
        """Apply the transform to the input image or mask.

        Args:
            input: the input tensor image or mask
            params: generated parameters
            flags: static parameters
            transform: the geometric transformation tensor

        Returns:
            the augmented input
        """
        directions = (params['direction'].round() * 2.0 - 1.0).to(
            torch.int
        )  # convert to -1 or 1
        dims = params['dim'].round().to(torch.int) + 2  # convert to 2 or 3
        sizes = torch.index_select(torch.tensor(input.shape), dim=0, index=dims)
        betas = params['beta']

        # Repeat each image gamma times (B*gamma, C, H, W)
        out = input.repeat_interleave(flags['gamma'], dim=0)

        # It's necessary to roll each image individually if shifts/dims vary
        # Apply roll to the i-th image along the chosen dimension
        # Note: We roll out[i] which has shape (C, H, W).
        # Because out[i] is a 3D tensor, we index using dim - 1 for torch.roll.
        for i, (beta, dim, direction, size) in enumerate(
            zip(betas, dims, directions, sizes, strict=True)
        ):
            shift = torch.round(beta * size * direction)
            out[i] = torch.roll(out[i], shifts=int(shift), dims=int(dim) - 1)
        return out
