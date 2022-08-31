"""Loss functions for learing on the prior."""

import torch
import torch.nn.functional as F
from torch.nn.modules import Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "torch.nn"


class QRLoss(Module):
    """The QR (forward) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    .. versionadded:: 0.2
    """

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

        q_log_p = torch.einsum("bcxy,bcxy->bxy", q, torch.log(target)).mean()

        loss = qbar_log_S - q_log_p
        return loss


class RQLoss(Module):
    """The RQ (backwards) loss between class probabilities and predictions.

    This loss is defined in `'Resolving label uncertainty with implicit generative
    models' <https://openreview.net/forum?id=AEa_UepnMDX>`_.

    .. versionadded:: 0.2
    """

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
        z = q / q.norm(  # type: ignore[no-untyped-call]
            p=1, dim=(0, 2, 3), keepdim=True
        ).clamp_min(1e-12).expand_as(q)
        r = F.normalize(z * target, p=1, dim=1)

        loss = torch.einsum("bcxy,bcxy->bxy", r, torch.log(r) - torch.log(q)).mean()

        return loss
