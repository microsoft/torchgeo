# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for temporal regression."""

from typing import Any, Literal

import einops
import torch
from torch import Tensor, nn

from ..datasets.utils import Sample
from ..models import LTAE, Presto, presto
from .base import BaseTask
from .mixins import RegressionMixin


class _PrestoTemporalRegressionModel(nn.Module):
    """Presto encoder with a regression head."""

    def __init__(self, model: Presto, out_features: int) -> None:
        """Initialize a new Presto regression model.

        Args:
            model: Presto model.
            out_features: Number of output features.
        """
        super().__init__()
        self.model = model
        self.head = nn.Linear(model.encoder.embedding_size, out_features)

    def forward(
        self,
        x: Tensor,
        dynamic_world: Tensor | None = None,
        latlons: Tensor | None = None,
        mask: Tensor | None = None,
        month: Tensor | int = 0,
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, T, C).
            dynamic_world: Dynamic world tensor of shape (B, T).
            latlons: Latitude and longitude tensor of shape (B, 2).
            mask: Mask tensor of shape (B, T, C).
            month: Month tensor or integer representing the month.

        Returns:
            Output tensor of shape (B, out_features).
        """
        b, t, _ = x.shape
        if dynamic_world is None:
            dynamic_world = torch.zeros(b, t, dtype=torch.long, device=x.device)
        if latlons is None:
            latlons = torch.zeros(b, 2, dtype=x.dtype, device=x.device)

        features, _, _ = self.model.encoder(
            x=x, dynamic_world=dynamic_world, latlons=latlons, mask=mask, month=month
        )
        return self.head(features.mean(dim=1))


class TemporalRegression(RegressionMixin, BaseTask):
    """Task for sequence-to-sequence temporal regression.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        model: Literal['ltae', 'presto'] = 'ltae',
        in_channels: int = 1,
        num_outputs: int = 1,
        labels: list[str] | None = None,
        out_steps: int = 1,
        loss: Literal['mae', 'mse'] = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a new TemporalRegression instance.

        Args:
            model: Name of the model architecture. Supported values are ``'ltae'``
                and ``'presto'``.
            in_channels: Number of input features per time step
                (the *C* dimension of the *(B, T, C)* input tensor).
            num_outputs: Number of output features per time step
                (the *C* dimension of the *(B, T, C)* target tensor).
            labels: List of feature names.
            out_steps: Number of output time steps
                (the *T* dimension of the *(B, T, C)* target tensor).
            loss: Loss function.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            **kwargs: Additional keyword arguments passed to the model constructor.
        """
        self.kwargs = kwargs
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        match self.hparams['model']:
            case 'ltae':
                out = self.hparams['num_outputs'] * self.hparams['out_steps']
                ltae = LTAE(in_channels=self.hparams['in_channels'], **self.kwargs)
                linear = nn.Linear(ltae.n_neurons[-1], out)
                self.model = nn.Sequential(ltae, linear)
            case 'presto':
                out = self.hparams['num_outputs'] * self.hparams['out_steps']
                model = presto(**self.kwargs)
                channels = sum(
                    len(group) for group in model.encoder.band_groups.values()
                )
                if self.hparams['in_channels'] != channels:
                    raise ValueError(
                        f'Presto expected {channels} input channels, got '
                        f'{self.hparams["in_channels"]}.'
                    )
                self.model = _PrestoTemporalRegressionModel(model, out)

    def _forward_model(self, batch: Sample) -> Tensor:
        """Forward batch inputs through the configured model.

        Args:
            batch: The output of the DataLoader.

        Returns:
            Predicted values of shape *(B, T * C)*.
        """
        x = batch['input']
        match self.hparams['model']:
            case 'ltae':
                y_hat: Tensor = self.model(x)
            case 'presto':
                kwargs: dict[str, Tensor] = {}
                for key in ['dynamic_world', 'latlons', 'mask', 'month']:
                    if key in batch:
                        kwargs[key] = batch[key]
                y_hat = self.model(x, **kwargs)

        return y_hat

    def _shared_step(self, batch: Sample, batch_idx: int, stage: str) -> Tensor:
        """Forward pass, loss computation, and metric update for all splits.

        Args:
            batch: The output of the DataLoader.
            batch_idx: Integer displaying index of this batch.
            stage: One of 'train', 'val', or 'test'.

        Returns:
            Scalar loss tensor.
        """
        y = batch['target']
        t = self.hparams['out_steps']
        batch_size = batch['input'].shape[0]

        y_hat = self._forward_model(batch)
        y_hat = einops.rearrange(y_hat, 'b (t c) -> b t c', t=t)

        loss = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss, batch_size=batch_size)

        # Denormalize before computing metrics
        datamodule = self.trainer.datamodule
        y = y * datamodule.target_std + datamodule.target_mean
        y_hat = y_hat * datamodule.target_std + datamodule.target_mean

        y = einops.rearrange(y, 'b t c -> (b t) c')
        y_hat = einops.rearrange(y_hat, 'b t c -> (b t) c')

        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat, y)

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
        t = self.hparams['out_steps']

        y_hat = self._forward_model(batch)
        y_hat = einops.rearrange(y_hat, 'b (t c) -> b t c', t=t)

        # Denormalize before returning predictions
        datamodule = self.trainer.datamodule
        y_hat = y_hat * datamodule.target_std + datamodule.target_mean

        return y_hat
