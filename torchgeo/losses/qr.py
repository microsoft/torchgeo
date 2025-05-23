# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Loss functions for learing on the prior."""

import torch
import torch.nn.functional as F
from torch.nn.modules import Module


class QRLoss(Module):
    """The QR (forward) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    .. versionadded:: 0.2
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize a new QRLoss instance.

        Args:
            eps: small constant for numerical stability to prevent division by zero
            and log(0) when computing the loss. Must be greater than or equal to 0.

        Raises:
            ValueError: If eps is less than 0.
        """
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')

        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the QR (forwards) loss on prior.

        Args:
            probs: probabilities of predictions, expected shape B x C x H x W.
            target: prior probabilities, expected shape B x C x H x W.

        Returns:
            qr loss
        """
        q = probs
        q_bar = q.mean(dim=(0, 2, 3))
        qbar_log_S = (q_bar * torch.log(q_bar)).sum()

        q_log_p = torch.einsum('bcxy,bcxy->bxy', q, torch.log(target + self.eps)).mean()

        loss = qbar_log_S - q_log_p
        return loss


class RQLoss(Module):
    """The RQ (backwards) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    .. versionadded:: 0.2
    """

    def __init__(self, eps: float = 1e-8) -> None:
        """Initialize a new RQLoss instance.

        Args:
            eps: small constant for numerical stability to prevent division by zero
            and log(0) when computing the loss. Must be greater than or equal to 0.

        Raises:
            ValueError: If eps is less than 0.
        """
        if not 0.0 <= eps:
            raise ValueError(f'Invalid epsilon value: {eps}')

        super().__init__()
        self.eps = eps

    def forward(self, probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the RQ (backwards) loss on prior.

        Args:
            probs: probabilities of predictions, expected shape B x C x H x W
            target: prior probabilities, expected shape B x C x H x W

        Returns:
            qr loss
        """
        q = probs

        # manually normalize due to https://github.com/pytorch/pytorch/issues/70100
        z = q / q.norm(p=1, dim=(0, 2, 3), keepdim=True).clamp_min(self.eps).expand_as(
            q
        )
        r = F.normalize(z * target, p=1, dim=1)

        loss = torch.einsum(
            'bcxy,bcxy->bxy', r, torch.log(r + self.eps) - torch.log(q + self.eps)
        ).mean()

        return loss
