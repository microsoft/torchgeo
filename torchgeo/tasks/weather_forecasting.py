# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for weather forecasting."""

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import Tensor, nn
from torchmetrics import MeanSquaredError, MetricCollection
from torchvision.models._api import WeightsEnum

from ..datasets.utils import Sample
from ..models import aurora_swin_unet
from .base import BaseTask


class WeatherForecasting(BaseTask):
    """Weather forecasting from prepared input and target states.

    Batches contain ``input_<variable>`` and ``target_<variable>`` fields. Surface
    tensors have shape ``(B, T, H, W)``, atmospheric tensors have shape
    ``(B, T, L, H, W)``, and static tensors have shape ``(H, W)``.

    The model receives input fields without the ``input_`` prefix and returns
    predicted variables under their dataset names with matching target shapes.

    .. versionadded:: 0.11
    """

    def __init__(
        self,
        data_vars: Sequence[str],
        model: str = 'aurora',
        weights: WeightsEnum | None = None,
        loss: Literal['mae', 'mse'] = 'mae',
        lr: float = 1e-4,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a new WeatherForecasting instance.

        Args:
            data_vars: List of data variables to predict.
            model: Weather forecasting model name. Supported value is ``'aurora'``.
            weights: Initial model weights.
            loss: Loss function, one of ``'mse'`` or ``'mae'``.
            lr: Learning rate for the optimizer.
            patience: Patience for the learning rate scheduler.
            **kwargs: Additional keyword arguments passed to the model constructor.
        """
        self.weights = weights
        self.kwargs = kwargs
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams['model']
        if model == 'aurora':
            self.model = aurora_swin_unet(
                weights=self.weights, data_vars=self.hparams['data_vars'], **self.kwargs
            )
        else:
            msg = f"Invalid model type '{model}'. Supported model: 'aurora'"
            raise ValueError(msg)

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        match self.hparams['loss']:
            case 'mse':
                self.criterion: nn.Module = nn.MSELoss()
            case 'mae':
                self.criterion = nn.L1Loss()

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        # dummy metrics, will change once we have latitude weighted version
        metrics = MetricCollection(
            {
                f'{name}_RMSE': MeanSquaredError(squared=False)
                for name in self.hparams['data_vars']
            },
            compute_groups=False,
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def forward(self, batch: Sample) -> Sample:
        """Predict weather states from the prepared inputs.

        Args:
            batch: Input and target states and their coordinates.

        Returns:
            Predicted tensors under their dataset variable names.
        """
        x = {
            name.removeprefix('input_'): value
            for name, value in batch.items()
            if name.startswith('input_')
        }
        y_hat: Sample = self.model(x)
        return y_hat

    def _shared_step(self, batch: Sample, stage: str) -> Tensor:
        """Compute the loss and metrics for a given stage.

        Raises:
            ValueError: If predictions and targets have different shapes.
        """
        batch_size = batch['input_time'].shape[0]
        predictions = self(batch)
        metrics = getattr(self, f'{stage}_metrics')
        losses: list[Tensor] = []
        for name in self.hparams['data_vars']:
            y_hat = predictions[name]
            y = batch[f'target_{name}'].to(y_hat)
            if y_hat.shape != y.shape:
                raise ValueError('Predictions and targets must have the same shape.')
            losses.append(self.criterion(y_hat, y))
            metrics[f'{stage}_{name}_RMSE'].update(y_hat.flatten(), y.flatten())
        loss = torch.stack(losses).mean()
        self.log(f'{stage}_loss', loss, batch_size=batch_size)
        return loss

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics."""
        return self._shared_step(batch, 'train')

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics."""
        self._shared_step(batch, 'val')

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics."""
        self._shared_step(batch, 'test')

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Sample:
        """Predict weather states from the prepared inputs."""
        return self(batch)

    def on_train_epoch_end(self) -> None:
        """Log train metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        """Log validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def on_test_epoch_end(self) -> None:
        """Log test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()
