# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Any, Literal, cast

import pytest
import torch

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.tasks import SpatioTemporalSegmentation


class TestSpatioTemporalSegmentation:
    @pytest.fixture(
        params=[
            pytest.param(
                ('convlstm', {'hidden_dim': 8, 'num_layers': 1}), id='convlstm'
            ),
            pytest.param(
                (
                    'utae',
                    {
                        'encoder_widths': (16, 16),
                        'decoder_widths': (8, 16),
                        'n_head': 4,
                        'd_model': 16,
                        'd_k': 4,
                    },
                ),
                id='utae',
            ),
        ]
    )
    def model_config(
        self, request: pytest.FixtureRequest
    ) -> tuple[Literal['convlstm', 'utae'], dict[str, Any]]:
        """Return spatiotemporal segmentation model configs."""
        return cast(tuple[Literal['convlstm', 'utae'], dict[str, Any]], request.param)

    @pytest.mark.parametrize(
        'name', ['pastis', 'pastis100', 'pastis_focal', 'pastis_jaccard']
    )
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
    def test_binary_task(
        self, model_config: tuple[Literal['convlstm', 'utae'], dict[str, Any]]
    ) -> None:
        model_name, kwargs = model_config
        model = SpatioTemporalSegmentation(
            model=model_name, in_channels=3, task='binary', loss='bce', **kwargs
        )
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'mask': torch.randint(0, 2, (2, 16, 16)),
            'length': torch.tensor([4, 4]),
            'batch_positions': torch.arange(4).repeat(2, 1),
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

    def test_multilabel_predict_step(
        self, model_config: tuple[Literal['convlstm', 'utae'], dict[str, Any]]
    ) -> None:
        model_name, kwargs = model_config
        model = SpatioTemporalSegmentation(
            model=model_name, in_channels=3, num_labels=4, task='multilabel', **kwargs
        )
        batch = {
            'image': torch.randn(2, 4, 3, 16, 16),
            'length': torch.tensor([4, 3]),
            'batch_positions': torch.arange(4).repeat(2, 1),
        }

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 4, 16, 16)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)

    def test_invalid_model(self) -> None:
        model = cast(Literal['convlstm', 'utae'], 'invalid')
        match = "Model type 'invalid' is not valid."

        with pytest.raises(ValueError, match=match):
            SpatioTemporalSegmentation(model=model)
