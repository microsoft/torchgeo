# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from types import SimpleNamespace
from typing import Literal

import pytest
import torch
from torch import Tensor

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.tasks import SpatioTemporalRegression


class ConstantRegressionModel(torch.nn.Module):
    """Model that returns zero-valued normalized predictions."""

    def forward(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        """Forward pass."""
        return torch.zeros(x.shape[0], 1, x.shape[-2], x.shape[-1], device=x.device)


class TestSpatioTemporalRegression:
    @pytest.mark.parametrize('name', ['copernicus_biomass_s3_ts'])
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

    @pytest.mark.parametrize('loss', ['mse', 'mae'])
    def test_task(self, loss: Literal['mse', 'mae']) -> None:
        model = SpatioTemporalRegression(
            in_channels=3, loss=loss, hidden_dim=8, num_layers=1
        )
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'mask': torch.rand(2, 16, 16),
            'length': torch.tensor([4, 3]),
        }
        try:
            model.training_step(batch, 0)
            model.validation_step(batch, 0)
            model.test_step(batch, 0)
        except MisconfigurationException:
            pass
        y_hat = model.predict_step(batch, 0)
        assert y_hat.shape == (2, 1, 16, 16)

    def test_predict_step_denormalizes_targets(self) -> None:
        model = SpatioTemporalRegression(
            in_channels=3, hidden_dim=8, num_layers=1
        )
        model.model = ConstantRegressionModel()
        datamodule = SimpleNamespace(
            target_mean=torch.tensor(2.0), target_std=torch.tensor(3.0)
        )
        setattr(model, '_trainer', SimpleNamespace(datamodule=datamodule))
        batch = {'image': torch.randn(2, 4, 3, 16, 16)}

        y_hat = model.predict_step(batch, 0)

        assert torch.all(y_hat == 2)

    def test_predict_step_without_target_stats(self) -> None:
        model = SpatioTemporalPixelwiseRegressionTask(
            in_channels=3, hidden_dim=8, num_layers=1
        )
        model.model = ConstantRegressionModel()
        batch = {'image': torch.randn(2, 4, 3, 16, 16)}

        y_hat = model.predict_step(batch, 0)

        assert torch.all(y_hat == 0)
