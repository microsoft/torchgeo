# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Loss functions for self-supervised learning."""

from typing import Union

import torch
import torch.nn.functional as F
from torch.nn import Module


class NTXentLoss(Module):
    """The Normalized-temperature Cross Entropy Loss (NT-Xent).

    This loss is defined in `'A Simple Framework for Contrastive Learning of
    Visual Representations' <https://arxiv.org/abs/2002.05709>`_.

    .. versionadded:: 0.5
    """

    def forward(
        self, z1: torch.Tensor, z2: torch.Tensor, t: Union[torch.Tensor, float]
    ) -> torch.Tensor:
        """Computes the NT-Xent loss.

        Args:
            z1: batch of vectors of shape (B, D).
            z2: batch of vectors of shape (B, D).
            t: temperature value to normalize the similarity vectors

        Returns:
            ntxent loss
        """
        batch_size = z1.shape[0]
        device = z1.device

        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=device)

        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        similarity = torch.matmul(z1, z2.T)
        similarity = similarity * torch.exp(t)
        targets = torch.arange(batch_size, device=device)
        loss = F.cross_entropy(similarity, targets)
        return loss
