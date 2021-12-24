"""Loss functions for learing on the prior."""

from typing import cast

import torch
import torch.nn.functional as F
from torch.nn.modules import Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


class QRLoss(Module):
    """The QR (forward) loss between class probabilities and predictions.

    This loss is defined in 'Resolving label uncertainty with implicit generative
    models' https://openreview.net/forum?id=AEa_UepnMDX.
    """

    def forward(self, log_probs: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Computes the QR (forwards) loss on prior.

        Args:
            log_probs: log-probabilities of predictions, expected shape B x C x H x W.
            target: prior probabilities, expected shape B x C x H x W.

        Returns:
            qr loss
        """
        q = torch.exp(log_probs)  # type: ignore[attr-defined]
        q_bar = q.mean(axis=(0, 2, 3))
        qbar_log_S = (q_bar * torch.log(q_bar)).sum()  # type: ignore[attr-defined]

        q_log_p = torch.einsum(  # type: ignore[attr-defined]
            "bcxy,bcxy->bxy", q, torch.log(target)  # type: ignore[attr-defined]
        ).mean()

        loss = qbar_log_S - q_log_p
        return cast(torch.Tensor, loss)


class RQLoss(Module):
    """The RQ (backwards) loss between class probabilities and predictions.

    This loss is defined in 'Resolving label uncertainty with implicit generative
    models' https://openreview.net/forum?id=AEa_UepnMDX.
    """

    def forward(
        self, log_probs: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Computes the RQ (backwards) loss on prior.

        Args:
            log_probs: log-probabilities of predictions, expected shape B x C x H x W
            target: prior probabilities, expected shape B x C x H x W

        Returns:
            qr loss
        """
        q = torch.exp(log_probs)  # type: ignore[attr-defined]

        # manually normalize due to https://github.com/pytorch/pytorch/issues/70100
        z = q / q.norm(p=1, dim=(0, 2, 3), keepdim=True).clamp_min(1e-12).expand_as(q)
        r = F.normalize(z * target, p=1, dim=1)

        loss = torch.einsum(  # type: ignore[attr-defined]
            "bcxy,bcxy->bxy",
            r,
            torch.log(r) - torch.log(q)  # type: ignore[attr-defined]
        ).mean()

        return cast(torch.Tensor, loss)
