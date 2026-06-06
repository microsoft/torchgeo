# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

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

    @pytest.mark.filterwarnings(r'ignore:You are trying to `self.log\(\)`')
    def test_binary_task(self) -> None:
        model = SpatioTemporalSegmentationTask(
            in_channels=3, task='binary', loss='bce', hidden_dim=8, num_layers=1
        )
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'mask': torch.randint(0, 2, (2, 16, 16)),
            'length': torch.tensor([4, 4]),
        }
        # Exercises y = y.float() for bce loss; self.log raises without a Trainer
        try:
            model.training_step(batch, 0)
        except MisconfigurationException:
            pass
        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 1, 16, 16)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)

    def test_multilabel_predict_step(self) -> None:
        model = SpatioTemporalSegmentationTask(
            in_channels=3, num_labels=4, task='multilabel', hidden_dim=8, num_layers=1
        )
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'length': torch.tensor([4, 3])}

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 4, 16, 16)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)
