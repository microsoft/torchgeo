# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Literal

import pytest
import torch

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.tasks import SpatioTemporalClassification


class TestSpatioTemporalClassificationTask:
    @pytest.mark.parametrize('name', ['quakeset_classification'])
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

    @pytest.mark.parametrize('task,num_classes', [('binary', 1), ('multiclass', 5)])
    def test_predict_step(
        self, task: Literal['binary', 'multiclass'], num_classes: int
    ) -> None:
        st_task = SpatioTemporalClassification(
            in_channels=4, task=task, num_classes=num_classes
        )
        # (B=2, T=3, C=4, H=16, W=16)
        batch = {'image': torch.randn(2, 3, 4, 16, 16)}
        prediction = st_task.predict_step(batch, 0)
        assert prediction.shape == (2, num_classes)

    def test_forward_shape(self) -> None:
        task = SpatioTemporalClassification(
            in_channels=10, task='multiclass', num_classes=20
        )
        x = torch.randn(2, 9, 10, 32, 32)
        y_hat = task(x)
        assert y_hat.shape == (2, 20)

    def test_binary_task(self) -> None:
        model = SpatioTemporalClassification(
            in_channels=3, task='binary', num_classes=1
        )
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'label': torch.randint(0, 2, (2,), dtype=torch.float),
            'length': torch.tensor([4, 4]),
        }
        # Exercises y = y.float() for bce loss; self.log raises without a Trainer
        try:
            model.training_step(batch, 0)
        except MisconfigurationException:
            pass
        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 1)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)
