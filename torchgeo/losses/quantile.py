# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Loss for quantile regression."""

from collections.abc import Sequence

import torch
from torch import Tensor, nn


class PinballLoss(nn.Module):
    """Pinball loss, averaged over quantiles, samples, and spatial dimensions.

    See equation (14) of `Image-to-Image Regression with Distribution-Free
    Uncertainty Quantification <https://arxiv.org/abs/2202.05265>`_.

    .. versionadded:: 0.11
    """

    quantiles: Tensor

    def __init__(self, quantiles: Sequence[float]) -> None:
        """Initialize a new PinballLoss instance.

        Args:
            quantiles: Quantile levels corresponding to prediction channels.

        Raises:
            ValueError: If quantiles are empty or outside (0, 1).
        """
        super().__init__()
        if not quantiles or any(not 0 < q < 1 for q in quantiles):
            raise ValueError('Quantiles must be nonempty and between 0 and 1.')
        self.register_buffer('quantiles', torch.tensor(quantiles))

    def forward(self, predictions: Tensor, target: Tensor) -> Tensor:
        """Compute the pinball loss.

        Args:
            predictions: Predictions of shape (B, Q, ...) for Q quantiles.
            target: Regression targets of shape (B, 1, ...).

        Returns:
            Mean pinball loss.

        Raises:
            ValueError: If prediction channels or target shape do not match.
        """
        if (
            predictions.ndim < 2
            or predictions.shape[1] != self.quantiles.numel()
            or target.shape != (predictions.shape[0], 1, *predictions.shape[2:])
        ):
            raise ValueError(
                'Expected predictions of shape (B, Q, ...) and target of shape '
                '(B, 1, ...), with Q matching the number of quantiles.'
            )
        quantiles = self.quantiles.to(dtype=predictions.dtype).view(
            1, -1, *([1] * (predictions.ndim - 2))
        )
        error = target - predictions
        return torch.maximum(quantiles * error, (quantiles - 1) * error).mean()
