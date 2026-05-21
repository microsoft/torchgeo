# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import pytest
import torch

from torchgeo.datamodules import MisconfigurationException
from torchgeo.trainers import SpatioTemporalPixelwiseRegressionTask


class TestSpatioTemporalPixelwiseRegressionTask:
    @pytest.mark.parametrize('loss', ['mse', 'mae'])
    def test_task(self, loss: str) -> None:
        model = SpatioTemporalPixelwiseRegressionTask(
            in_channels=3, loss=loss, hidden_dim=8, num_layers=1
        )
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'mask': torch.rand(2, 16, 16)}
        try:
            model.training_step(batch, 0)
        except MisconfigurationException:
            pass
        y_hat = model.predict_step(batch, 0)
        assert y_hat.shape == (2, 1, 16, 16)

    def test_predict_step_with_lengths(self) -> None:
        model = SpatioTemporalPixelwiseRegressionTask(
            in_channels=3, hidden_dim=8, num_layers=1
        )
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'length': torch.tensor([4, 3])}
        y_hat = model.predict_step(batch, 0)
        assert y_hat.shape == (2, 1, 16, 16)
