# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.trainers import SpatioTemporalSegmentationTask


class TestSpatioTemporalSegmentationTask:
    @pytest.mark.parametrize('name', ['pastis', 'pastis_focal', 'pastis_jaccard'])
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

    def _make_task(self, **kwargs: Any) -> SpatioTemporalSegmentationTask:
        task = SpatioTemporalSegmentationTask(hidden_dim=8, num_layers=1, **kwargs)
        # Avoid Lightning warnings when calling step hooks without a Trainer.
        task.log = MagicMock()  # type: ignore[method-assign]
        task.log_dict = MagicMock()  # type: ignore[method-assign]
        return task

    def test_convlstm_timeseries_forward_and_step(self) -> None:
        model = SpatioTemporalSegmentationTask(
            model='convlstm',
            in_channels=10,
            num_classes=5,
            task='multiclass',
            hidden_dim=8,
            num_layers=1,
        )
        batch = {
            'image': torch.randn(2, 7, 10, 16, 16),
            'mask': torch.randint(0, 5, (2, 16, 16)),
            'length': torch.tensor([7, 5]),
        }
        y_hat = model(batch['image'], lengths=batch['length'])
        assert y_hat.shape == (2, 5, 16, 16)

        # If no lengths are provided, the model uses the last timestep.
        # This should match the explicit `lengths=T` case.
        y_hat_no_lengths = model(batch['image'])
        y_hat_last_step = model(batch['image'], lengths=torch.tensor([7, 7]))
        torch.testing.assert_close(y_hat_no_lengths, y_hat_last_step)

        # Lengths longer than the available sequence should clamp to the
        # final timestep instead of indexing out of bounds.
        y_hat_clamped = model(batch['image'], lengths=torch.tensor([9.0, 12.0]))
        torch.testing.assert_close(y_hat_no_lengths, y_hat_clamped)

    def test_binary_steps_and_predict_step(self) -> None:
        model = self._make_task(in_channels=3, task='binary', loss='bce')
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'mask': torch.randint(0, 2, (2, 16, 16)),
            'length': torch.tensor([4, 2]),
        }

        train_loss = model.training_step(batch, 0)
        assert train_loss.ndim == 0

        assert model.validation_step(batch, 0) is None
        assert model.test_step(batch, 0) is None

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 1, 16, 16)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)

    def test_multiclass_predict_step(self) -> None:
        model = self._make_task(in_channels=3, num_classes=4, task='multiclass')
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'length': torch.tensor([4, 3])}

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 4, 16, 16)
        torch.testing.assert_close(
            probabilities.sum(dim=1), torch.ones((2, 16, 16)), atol=1e-5, rtol=1e-5
        )
