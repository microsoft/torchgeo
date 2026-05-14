# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
from collections.abc import Callable
from typing import Any

import pytest
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

from torchgeo.datamodules import MisconfigurationException
from torchgeo.main import main
from torchgeo.models import ConvLSTM
from torchgeo.trainers import SpatioTemporalSegmentationTask


class TestSpatioTemporalSegmentationTask:
    def test_trainer_with_pastis_config(self, fast_dev_run: bool) -> None:
        config = os.path.join('tests', 'conf', 'pastis.yaml')

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

    @pytest.fixture
    def create_spatiotemporal_model(
        self,
    ) -> Callable[..., SpatioTemporalSegmentationTask]:
        def _create_spatiotemporal_model(
            **kwargs: Any,
        ) -> SpatioTemporalSegmentationTask:
            model = SpatioTemporalSegmentationTask(hidden_dim=8, num_layers=1, **kwargs)
            # Avoid Lightning warnings when calling step hooks without a Trainer.
            setattr(model, 'log', lambda *args, **kwargs: None)
            setattr(model, 'log_dict', lambda *args, **kwargs: None)
            return model

        return _create_spatiotemporal_model

    def test_spatiotemporal_forward_defaults_to_convlstm(self) -> None:
        model = SpatioTemporalSegmentationTask(in_channels=3, num_classes=5)
        assert model.hparams['model'] == 'convlstm'
        assert isinstance(model.model, ConvLSTM)
        assert model.model.head is not None

    def test_spatiotemporal_forward_supports_direct_convlstm_kwargs(self) -> None:
        model = SpatioTemporalSegmentationTask(
            in_channels=3, num_classes=5, hidden_dim=8, num_layers=1
        )
        assert isinstance(model.model, ConvLSTM)
        assert model.model.hidden_dim == [8]

    def test_spatiotemporal_invalid_model(self) -> None:
        invalid_model: Any = 'invalid'
        match = "Invalid model type 'invalid'. Supported model: 'convlstm'"
        with pytest.raises(ValueError, match=match):
            SpatioTemporalSegmentationTask(model=invalid_model)

    def test_spatiotemporal_direct_kwargs_are_saved_in_hparams(self) -> None:
        model = SpatioTemporalSegmentationTask(
            in_channels=3, num_classes=5, hidden_dim=8, num_layers=1
        )

        assert model.hparams['hidden_dim'] == 8
        assert model.hparams['num_layers'] == 1
        assert 'kwargs' not in model.hparams

    def test_convlstm_timeseries_forward_and_step(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(
            model='convlstm', in_channels=10, num_classes=5, task='multiclass'
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

        loss = model.training_step(batch, 0)
        assert loss.ndim == 0

    def test_ce_class_weights_from_sequence(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(
            in_channels=3, num_classes=2, task='multiclass', class_weights=[1.0, 2.0]
        )

        assert isinstance(model.criterion, nn.CrossEntropyLoss)
        torch.testing.assert_close(
            model.criterion.weight, torch.tensor([1.0, 2.0], dtype=torch.float32)
        )

    @pytest.mark.parametrize(
        ('loss', 'expected_type'),
        [('jaccard', smp.losses.JaccardLoss), ('focal', smp.losses.FocalLoss)],
    )
    def test_alternate_losses(
        self,
        create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask],
        loss: str,
        expected_type: type[nn.Module],
    ) -> None:
        model = create_spatiotemporal_model(
            in_channels=3, num_classes=3, task='multiclass', loss=loss, ignore_index=1
        )

        assert isinstance(model.criterion, expected_type)

    def test_binary_steps_and_predict_step(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(in_channels=3, task='binary', loss='bce')
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

    def test_multiclass_predict_step(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(
            in_channels=3, num_classes=4, task='multiclass'
        )
        batch = {'image': torch.randn(2, 4, 3, 16, 16), 'length': torch.tensor([4, 3])}

        probabilities = model.predict_step(batch, 0)
        assert probabilities.shape == (2, 4, 16, 16)
        torch.testing.assert_close(
            probabilities.sum(dim=1), torch.ones((2, 16, 16)), atol=1e-5, rtol=1e-5
        )

    def test_multiclass_classwise_metrics(
        self, create_spatiotemporal_model: Callable[..., SpatioTemporalSegmentationTask]
    ) -> None:
        model = create_spatiotemporal_model(
            in_channels=3,
            num_classes=3,
            task='multiclass',
            labels=['background', 'crop', 'water'],
        )
        y_hat = torch.randn(2, 3, 16, 16)
        y = torch.randint(0, 3, (2, 16, 16))

        model.val_metrics(y_hat, y)
        metrics = model.val_metrics.compute()

        assert 'val_OverallAccuracy' in metrics
        assert 'val_AverageJaccardIndex' in metrics
        assert 'val_ClasswiseAccuracy_background' in metrics
        assert 'val_ClasswiseAccuracy_crop' in metrics
        assert 'val_ClasswiseAccuracy_water' in metrics
        assert 'val_ClasswiseJaccardIndex_background' in metrics
