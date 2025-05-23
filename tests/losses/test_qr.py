# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch
import torch.nn.functional as F

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

    def test_invalid_eps_value(self) -> None:
        """Test that the QR loss raises a ValueError when eps is less than 0."""
        with pytest.raises(ValueError, match='Invalid epsilon value'):
            QRLoss(eps=-1.0)

        with pytest.raises(ValueError, match='Invalid epsilon value'):
            RQLoss(eps=-1.0)

    def test_eps_prevents_nan(self) -> None:
        """Test that the QR loss does not produce NaN when eps is set to a small value."""
        probs = torch.rand(2, 4, 10, 10).softmax(dim=1)
        labels = torch.randint(0, 4, (2, 10, 10))
        targets = F.one_hot(labels, num_classes=4).float().permute(0, 3, 1, 2)

        loss = QRLoss(eps=1e-8)(probs, targets)
        assert not torch.isnan(loss).any(), 'Loss should not contain NaN values'

        loss = RQLoss(eps=1e-8)(probs, targets)
        assert not torch.isnan(loss).any(), 'Loss should not contain NaN values'
