# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Focal Jaccard loss functions."""

from typing import cast

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class BinaryFocalJaccardLoss(nn.Module):
    """Binary Focal Jaccard Loss."""

    def __init__(self) -> None:
        """Initialize a BinaryFocalJaccardLoss instance."""
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(mode='binary', normalized=True)
        self.jaccard_loss = smp.losses.JaccardLoss(mode='binary')

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the loss."""
        return cast(
            torch.Tensor,
            self.focal_loss(preds, targets) + self.jaccard_loss(preds, targets),
        )
