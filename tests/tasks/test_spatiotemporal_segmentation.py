# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import pytest
import torch
from torch import Tensor

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.tasks import SpatioTemporalSegmentation


class TestSpatioTemporalSegmentation:
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

    def test_binary_task(self) -> None:
        model = SpatioTemporalSegmentation(
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
        model = SpatioTemporalSegmentation(
            in_channels=3, num_labels=4, task='multilabel', hidden_dim=8, num_layers=1
        )
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'length': torch.tensor([4, 3])}

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 4, 16, 16)
        assert torch.all(probabilities >= 0)
        assert torch.all(probabilities <= 1)

    def test_utae_predict_step_forwards_batch_positions(self) -> None:
        model = SpatioTemporalSegmentationTask(
            model='utae',
            in_channels=3,
            num_classes=4,
            encoder_widths=(4, 4),
            decoder_widths=(4, 4),
            out_conv=(4, 4),
            n_head=1,
            d_model=4,
            d_k=4,
        )
        captured: list[Tensor] = []

        # Hook the nested temporal encoder to verify the trainer forwards
        # batch_positions while still exercising the real UTAE model.
        def hook(
            module: torch.nn.Module,
            args: tuple[object, ...],
            kwargs: dict[str, object],
            output: object,
        ) -> None:
            """Record forwarded batch positions."""
            batch_positions_kwarg = kwargs['batch_positions']
            assert isinstance(batch_positions_kwarg, Tensor)
            captured.append(batch_positions_kwarg)

        handle = model.model.temporal_encoder.register_forward_hook(
            hook, with_kwargs=True
        )
        batch_positions = torch.tensor([[1, 2, 3], [4, 5, 6]])
        batch = {
            'image': torch.randn(2, 3, 3, 16, 16),
            'batch_positions': batch_positions,
        }

        try:
            probabilities = model.predict_step(batch, 0)
        finally:
            handle.remove()

        assert captured[0] is batch_positions
        assert probabilities.shape == (2, 4, 16, 16)
