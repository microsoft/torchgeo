# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch

from torchgeo.losses import NTXentLoss


class TestSSLLosses:
    def test_ntxent_loss(self) -> None:
        z1 = torch.randn(2, 8)
        z2 = torch.randn(2, 8)
        t = torch.tensor(0.5)
        loss = NTXentLoss()(z1, z2, t)
        assert loss.ndim == 0
        assert loss.dtype == torch.float
