# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.losses import QRLoss, RQLoss


class TestQRLosses:
    def test_loss_on_prior_simple(self) -> None:
        probs = torch.rand(2, 4, 10, 10)
        log_probs = torch.log(probs)
        targets = torch.rand(2, 4, 10, 10)
        QRLoss()(log_probs, targets)

    def test_loss_on_prior_reversed_kl_simple(self) -> None:
        probs = torch.rand(2, 4, 10, 10)
        log_probs = torch.log(probs)
        targets = torch.rand(2, 4, 10, 10)
        RQLoss()(log_probs, targets)
