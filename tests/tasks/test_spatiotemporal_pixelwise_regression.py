# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from lightning.pytorch import Trainer

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.tasks import SpatioTemporalPixelwiseRegression


class TestSpatioTemporalPixelwiseRegression:
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

    def test_predict_step(self) -> None:
        model = SpatioTemporalPixelwiseRegression(
            in_channels=3, hidden_dim=8, num_layers=1
        )
        model._trainer = cast(
            Trainer,
            SimpleNamespace(
                datamodule=SimpleNamespace(target_mean=2.0, target_std=3.0)
            ),
        )
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'length': torch.tensor([4, 3])}

        normalized = model(batch['image'], lengths=batch['length'])
        predictions = model.predict_step(batch, 0)

        assert predictions.shape == (2, 1, 16, 16)
        assert torch.allclose(predictions, normalized * 3 + 2)
