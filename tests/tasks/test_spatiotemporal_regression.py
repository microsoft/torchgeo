# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Literal

import pytest
import torch

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.tasks import SpatioTemporalRegression

# Temporarily set target_key to 'magnitude'
SpatioTemporalRegression.target_key = 'magnitude'

pytest.importorskip('h5py', minversion='3.10')


class TestSpatioTemporalRegressionTask:
    @pytest.mark.parametrize('name', ['quakeset_regression'])
    def test_trainer(self, name: str, fast_dev_run: bool) -> None:
        config = os.path.join('tests', 'conf', name + '.yaml')

        args = [
            '--config',
            config,
            '--trainer.accelerator',
            'cpu',
            '--trainer.fast_dev_run',
            str(fast_dev_run),
            '--trainer.max_epochs',
            '1',
            '--trainer.log_every_n_steps',
            '1',
        ]

        main(['fit', *args])
        try:
            main(['test', *args])
        except MisconfigurationException:
            pass
        try:
            main(['predict', *args])
        except MisconfigurationException:
            pass

    def test_predict_step(self) -> None:
        st_task = SpatioTemporalRegression(in_channels=4)
        # (B=2, T=3, C=4, H=16, W=16)
        batch = {'image': torch.randn(2, 3, 4, 16, 16)}
        prediction = st_task.predict_step(batch, 0)
        assert prediction.shape == (2, 1)

    def test_forward_shape(self) -> None:
        task = SpatioTemporalRegression(in_channels=10, num_outputs=20)
        x = torch.randn(2, 9, 10, 32, 32)
        y_hat = task(x)
        assert y_hat.shape == (2, 20)

    @pytest.mark.parametrize('loss', ['mse', 'mae'])
    def test_losses(self, loss: Literal['mse', 'mae']) -> None:
        task = SpatioTemporalRegression(in_channels=3, loss=loss)
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'magnitude': torch.randn(2),
            'length': torch.tensor([4, 4]),
        }
        try:
            task.training_step(batch, 0)
        except MisconfigurationException:
            pass
