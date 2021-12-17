# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.losses import loss_on_prior_reversed_kl_simple, loss_on_prior_simple


class TestQRLosses:
    def test_loss_on_prior_simple(self) -> None:
        probs = torch.rand(2, 4, 10, 10)
        log_probs = torch.log(probs)  # type: ignore[attr-defined]
        targets = torch.rand(2, 4, 10, 10)
        loss_on_prior_simple(log_probs, targets)

    def test_loss_on_prior_reversed_kl_simple(self) -> None:
        probs = torch.rand(2, 4, 10, 10)
        log_probs = torch.log(probs)  # type: ignore[attr-defined]
        targets = torch.rand(2, 4, 10, 10)
        loss_on_prior_reversed_kl_simple(log_probs, targets)
