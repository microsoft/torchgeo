# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo color transforms."""

from kornia.augmentation import IntensityAugmentationBase2D
from torch import Tensor


class RandomGrayscale(IntensityAugmentationBase2D):
    r"""Apply random transformation to grayscale according to a probability p value.

    There is no single agreed upon definition of grayscale for MSI. Some possibilities
    include:

    * Average of all bands: :math:`\frac{1}{C}` where :math:`C` is the number of
      spectral channels.
    * RGB-only bands: :math:`[0.299, 0.587, 0.114]` for the RGB channels, 0 for
      all other channels.
    * PCA: the first principal component across the spectral axis computed via PCA,
      minimizes redundant information.

    The weight vector you provide will be automatically rescaled to sum to 1 in order
    to avoid changing the intensity of the image.

    .. versionadded:: 0.5
    """

    def __init__(
        self,
        weights: Tensor,
        p: float = 0.1,
        same_on_batch: bool = False,
        keepdim: bool = False,
    ) -> None:
        """Initialize a new RandomGrayscale instance.

        Args:
            weights: Weights applied to each channel to compute a grayscale
                representation. Should be the same length as the number of channels.
            p: Probability of the image to be transformed to grayscale.
            same_on_batch: Apply the same transformation across the batch.
            keepdim: Whether to keep the output shape the same as input (True)
                or broadcast it to the batch form (False).
        """
        super().__init__(p=p, same_on_batch=same_on_batch, keepdim=keepdim)

        # Rescale to sum to 1
        weights /= weights.sum()

        self.flags = {'weights': weights}

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
            The augmented input.
        """
        weights = flags['weights'][..., :, None, None].to(input.device)
        out = input * weights
        out = out.sum(dim=-3)
        out = out.unsqueeze(-3).expand(input.shape)
        return out
