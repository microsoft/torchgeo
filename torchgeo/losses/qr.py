# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Loss functions for learing on the prior."""

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Module


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
        # https://github.com/pytorch/pytorch/issues/116327
        q_bar: Tensor = q.mean(dim=(0, 2, 3))
        log_q_bar = torch.log(q_bar)
        qbar_log_S: Tensor = q_bar * log_q_bar
        qbar_log_S = qbar_log_S.sum()

        q_log_p = torch.einsum("bcxy,bcxy->bxy", q, torch.log(target))
        q_log_p = q_log_p.mean()

        loss: Tensor = qbar_log_S - q_log_p
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
        z = q / q.norm(p=1, dim=(0, 2, 3), keepdim=True).clamp_min(1e-12).expand_as(q)
        r = F.normalize(z * target, p=1, dim=1)

        loss = torch.einsum("bcxy,bcxy->bxy", r, torch.log(r) - torch.log(q))
        loss = loss.mean()

        return loss
