# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Cross-Entropy Jaccard loss functions."""

from typing import cast

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class BinaryXEntJaccardLoss(nn.Module):
    """Binary Cross-Entropy Jaccard Loss."""

    def __init__(self) -> None:
        """Initialize a BinaryXEntJaccardLoss instance."""
        super().__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.jaccard_loss = smp.losses.JaccardLoss(mode='binary')

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss."""
        return cast(
            torch.Tensor,
            self.bce_loss(preds, targets) + self.jaccard_loss(preds, targets),
        )
