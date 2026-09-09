# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.losses import PinballLoss


class TestPinballLoss:
    @pytest.mark.parametrize('shape', [(2, 3), (2, 3, 2, 4)])
    def test_forward(self, shape: tuple[int, ...]) -> None:
        predictions = torch.tensor([[0.0, 3.0, 4.0], [2.0, 0.0, -1.0]])
        gradient = torch.tensor([[-0.2, 0.5, 0.2], [0.8, -0.5, -0.8]])
        if len(shape) == 4:
            predictions = predictions[:, :, None, None].expand(shape).clone()
            gradient = gradient[:, :, None, None].expand(shape)
        predictions.requires_grad_()
        target = torch.ones((shape[0], 1, *shape[2:]))

        loss = PinballLoss([0.2, 0.5, 0.8])(predictions, target)
        torch.testing.assert_close(loss, torch.tensor(4.7 / 6))
        loss.backward()
        torch.testing.assert_close(predictions.grad, gradient / predictions.numel())

    @pytest.mark.parametrize('quantiles', [[], [0], [1], [-0.1], [1.1], [float('nan')]])
    def test_invalid_quantiles(self, quantiles: list[float]) -> None:
        with pytest.raises(ValueError, match='nonempty and between 0 and 1'):
            PinballLoss(quantiles)
