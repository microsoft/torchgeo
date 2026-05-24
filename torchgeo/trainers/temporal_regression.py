# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for temporal regression."""

from typing import Any, Literal

import einops
from torch import Tensor, nn
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from ..datasets.utils import Sample
from ..models import LTAE
from .base import BaseTask


class TemporalRegressionTask(BaseTask):
    """Trainer for sequence-to-sequence temporal regression.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        model: Literal['ltae'] = 'ltae',
        in_features: int = 1,
        out_features: int = 1,
        out_steps: int = 1,
        loss: Literal['mae', 'mse'] = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a new TemporalRegressionTask instance.

        Args:
            model: Name of the model architecture.
            in_features: Number of input features per time step
                (the *C* dimension of the *(B, T, C)* input tensor).
            out_features: Number of output features per time step
                (the *C* dimension of the *(B, T, C)* target tensor).
            out_steps: Number of output time steps
                (the *T* dimension of the *(B, T, C)* target tensor).
            loss: Loss function.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            **kwargs: Additional keyword arguments passed to the model constructor.

        .. versionadded:: 0.10
        """
        self.kwargs = kwargs
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        match self.hparams['model']:
            case 'ltae':
                out = self.hparams['out_features'] * self.hparams['out_steps']
                ltae = LTAE(in_channels=self.hparams['in_features'], **self.kwargs)
                linear = nn.Linear(ltae.n_neurons[-1], out)
                self.model = nn.Sequential(ltae, linear)

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        match self.hparams['loss']:
            case 'mse':
                self.criterion: nn.Module = nn.MSELoss()
            case 'mae':
                self.criterion = nn.L1Loss()

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.MeanSquaredError` (``MSE``) and its square root
          (``RMSE``). Lower is better.
        * :class:`~torchmetrics.MeanAbsoluteError` (``MAE``). Lower is better.
        """
        metrics = MetricCollection(
            {
                'RMSE': MeanSquaredError(squared=False),
                'MSE': MeanSquaredError(squared=True),
                'MAE': MeanAbsoluteError(),
            }
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _shared_step(self, batch: Sample, batch_idx: int, stage: str) -> Tensor:
        """Forward pass, loss computation, and metric update for all splits.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            stage: One of 'train', 'val', or 'test'.

        Returns:
            Scalar loss tensor.
        """
        x = batch['input']
        y = batch['target']
        t = self.hparams['out_steps']
        batch_size = x.shape[0]

        y_hat = self.model(x)
        y_hat = einops.rearrange(y_hat, 'b (t c) -> b t c', t=t)

        loss = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss, batch_size=batch_size)

        # Denormalize before computing metrics
        datamodule = self.trainer.datamodule
        y = y * datamodule.target_mean + datamodule.target_std
        y_hat = y_hat * datamodule.target_mean + datamodule.target_std

        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat, y)
        self.log_dict(metrics, batch_size=batch_size)

        return loss

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        return self._shared_step(batch, batch_idx, 'train')

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._shared_step(batch, batch_idx, 'test')

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute predicted values.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Predicted values of shape *(B, T, C)*.
        """
        x = batch['input']
        t = self.hparams['out_steps']

        y_hat = self.model(x)
        y_hat = einops.rearrange(y_hat, 'b (t c) -> b t c', t=t)

        # Denormalize before returning predictions
        datamodule = self.trainer.datamodule
        y_hat = y_hat * datamodule.target_mean + datamodule.target_std

        return y_hat
