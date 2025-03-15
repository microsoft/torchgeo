# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for autoregression."""

from typing import Any

import torch.nn as nn
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from torchgeo.models import LSTMSeq2Seq

from .base import BaseTask


class AutoregressionTask(BaseTask):
    """Autoregression."""

    def __init__(
        self,
        model: str = 'lstm_seq2seq',
        input_size: int = 1,
        input_size_decoder: int = 1,
        hidden_size: int = 1,
        output_size: int = 1,
        target_indices: list[int] | None = None,
        encoder_indices: list[int] | None = None,
        decoder_indices: list[int] | None = None,
        timesteps_ahead: int = 1,
        num_layers: int = 1,
        loss: str = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
        teacher_force_prob: float | None = None,
    ) -> None:
        """Initialize a new AutoregressionTask instance.

        Args:
            model: Name of the model to use, currently supports 'lstm_seq2seq'.
                Defaults to 'lstm_seq2seq'.
            input_size: The number of features in the input. Defaults to 1.
            input_size_decoder: The number of features in the decoder input.
                Defaults to 1.
            hidden_size: The number of features in the hidden states of the encoder
                and decoder. Defaults to 1.
            output_size: The number of features output by the model. Defaults to 1.
            target_indices: The indices of the target(s) in the dataset. If None, uses all features. Defaults to None.
            encoder_indices: The indices of the encoder inputs. If None, uses all features. Defaults to None.
            decoder_indices: The indices of the decoder inputs. If None, uses all features. Defaults to None.
            timesteps_ahead: Number of time steps to predict. Defaults to 1.
            num_layers: Number of LSTM layers in the encoder and decoder. Defaults to 1.
            loss: One of 'mse' or 'mae'. Defaults to 'mse'.
            lr: Learning rate for optimizer. Defaults to 1e-3.
            patience: Patience for learning rate scheduler. Defaults to 10.
            teacher_force_prob: Probability of using teacher forcing. If None, does not
                use teacher forcing. Defaults to None.
        """
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        model: str = self.hparams['model']
        input_size = self.hparams['input_size']
        input_size_decoder = self.hparams['input_size_decoder']
        hidden_size = self.hparams['hidden_size']
        output_size = self.hparams['output_size']
        timesteps_ahead = self.hparams['timesteps_ahead']
        num_layers = self.hparams['num_layers']
        target_indices = self.hparams['target_indices']
        encoder_indices = self.hparams['encoder_indices']
        decoder_indices = self.hparams['decoder_indices']
        teacher_force_prob = self.hparams['teacher_force_prob']

        if model == 'lstm_seq2seq':
            self.model = LSTMSeq2Seq(
                input_size_encoder=input_size,
                input_size_decoder=input_size_decoder,
                target_indices=target_indices,
                encoder_indices=encoder_indices,
                decoder_indices=decoder_indices,
                hidden_size=hidden_size,
                output_size=output_size,
                output_sequence_len=timesteps_ahead,
                num_layers=num_layers,
                teacher_force_prob=teacher_force_prob,
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'lstm_seq2seq'."
            )

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams['loss']
        if loss == 'mse':
            self.criterion: nn.Module = nn.MSELoss()
        elif loss == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'mse' or 'mae' loss."
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
        target_indices = self.hparams['target_indices']
        past_steps, future_steps = batch
        y_hat = self(past_steps, future_steps)
        if target_indices:
            future_steps = future_steps[:, :, target_indices]
        loss: Tensor = self.criterion(y_hat, future_steps)
        self.log(f'{stage}_loss', loss)

        # Retrieve the correct metrics based on the stage
        metrics = getattr(self, f'{stage}_metrics', None)
        if metrics:
            metrics(y_hat, future_steps)
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
        past_steps, future_steps = batch
        y_hat = self(past_steps, future_steps)
        mean = past_steps.mean(dim=0, keepdim=True)
        std = past_steps.std(dim=0, keepdim=True)
        y_hat_denormalize: Tensor = y_hat * std + mean
        return y_hat_denormalize
