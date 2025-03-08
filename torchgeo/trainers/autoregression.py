# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for autoregression."""

from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError
from torchvision.models import LSTMSeq2Seq
from torchvision.models._api import WeightsEnum

from .base import BaseTask


class AutoregressionTask(BaseTask):
    """Autoregression."""

    def __init__(
        self,
        model: str = 'lstm',
        weights: WeightsEnum | str | bool | None = None,
        input_size: int = 1,
        input_size_decoder: int = 1,
        hidden_size: int = 1,
        output_size: int = 1,
        lookback: int = 3,
        timesteps_ahead: int = 1,
        num_layers: int = 1,
        loss: str = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
    ) -> None:
        """Initialize a new AutoregressionTask instance.

        Args:
            model: Name of the model to use, currently supports 'lstm' or 'seq2seq'.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            loss: One of 'mse' or 'mae'.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.

        .. versionadded: 0.7
        """
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        model: str = self.hparams['model']
        input_size = self.hparams['input_size']
        input_size_decoder = self.hparams['input_size_decoder']
        hidden_size = self.hparams['hidden_size']
        output_size = self.hparams['output_size']
        lookback = self.hparams['lookback']
        timesteps_ahead = self.hparams['timesteps_ahead']
        num_layers = self.hparams['num_layers']

        if model == 'lstm':
            assert timesteps_ahead == 1, (
                f'LSTM only supports 1 timestep ahead, got timesteps_ahead={timesteps_ahead}.'
            )
            self.model = torch.nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
            )
        elif model == 'seq2seq':
            self.model = LSTMSeq2Seq(
                input_size_encoder=input_size,
                input_size_decoder=input_size_decoder,
                hidden_size=hidden_size,
                output_size=output_size,
                output_seq_length=timesteps_ahead,
                num_layers=num_layers,
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'lstm' and 'seq2seq'."
            )

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams['loss']
        if loss == 'mse':
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. Currently, supports 'mse' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        output_size = self.hparams['output_size']
        metrics = MetricCollection(
            {
                'rmse': MeanSquaredError(num_outputs=output_size, squared=False),
                'mae': MeanAbsoluteError(num_outputs=output_size),
            }
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        """Compute the loss and additional metrics for the given stage.

        Args:
            batch: The output of your DataLoader._
            batch_idx: Integer displaying index of this batch._
            stage: The current stage.

        Returns:
            The loss tensor.
        """
        x, y = batch
        y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss)

        # Retrieve the correct metrics based on the stage
        metrics = getattr(self, f'{stage}_metrics', None)
        if metrics:
            metrics(y_hat, y)
            self.log_dict({f'{k}': v for k, v in metrics.compute().items()})

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.

        Returns:
            The loss tensor.
        """
        loss = self._shared_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
        """
        self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
        """
        self._shared_step(batch, batch_idx, 'test')

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted regression values.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted values.
        """
        x = batch
        y_hat: Tensor = self(x)
        return y_hat
