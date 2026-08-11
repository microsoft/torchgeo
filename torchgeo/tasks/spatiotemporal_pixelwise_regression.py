# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for spatiotemporal pixelwise regression."""

from typing import Any, Literal

import torch
from torch import Tensor

from ..datasets.utils import Sample
from ..models import ConvLSTM
from .base import BaseTask
from .mixins import RegressionMixin


class SpatioTemporalPixelwiseRegression(RegressionMixin, BaseTask):
    """Pixelwise regression over spatiotemporal image sequences.

    Uses a :class:`~torchgeo.models.ConvLSTM` encoder to predict continuous
    per-pixel values from images of shape ``(B, T, C, H, W)``.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        model: Literal['convlstm'] = 'convlstm',
        in_channels: int = 3,
        num_outputs: int = 1,
        labels: list[str] | None = None,
        loss: Literal['mse', 'mae'] = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SpatioTemporalPixelwiseRegression instance.

        Args:
            model: Spatiotemporal model name. Supported value is ``'convlstm'``.
            in_channels: Number of channels per timestep for inputs of shape
                ``(B, T, C, H, W)``.
            num_outputs: Number of output channels.
            labels: List of output channel names.
            loss: Loss function, one of ``'mse'`` or ``'mae'``.
            lr: Learning rate for the optimizer.
            patience: Patience for the learning-rate scheduler.
            **kwargs: Additional keyword arguments passed to the model constructor.
        """
        self.kwargs = kwargs
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        self.model = ConvLSTM(
            input_dim=self.hparams['in_channels'],
            num_classes=self.hparams['num_outputs'],
            **self.kwargs,
        )

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape ``(B, T, C, H, W)``.
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            Output tensor of shape ``(B, C, H, W)``.
        """
        return self.model(x, **kwargs)

    def _shared_step(self, batch: Sample, stage: str) -> Tensor:
        """Compute the loss and metrics for a given stage."""
        x = batch['image']
        y = batch['mask'].to(torch.float)
        batch_size = x.shape[0]

        kwargs: dict[str, Tensor] = {}
        if (lengths := batch.get('length')) is not None:
            kwargs['lengths'] = lengths

        y_hat = self(x, **kwargs).squeeze(dim=1)
        loss: Tensor = self.criterion(y_hat, y)

        datamodule = self.trainer.datamodule
        y = y * datamodule.target_std + datamodule.target_mean
        y_hat = y_hat * datamodule.target_std + datamodule.target_mean
        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat, y)

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
    ) -> Tensor:
        """Compute the predicted regression values."""
        kwargs: dict[str, Tensor] = {}
        if (lengths := batch.get('length')) is not None:
            kwargs['lengths'] = lengths
        y_hat: Tensor = self(batch['image'], **kwargs)
        datamodule = self.trainer.datamodule
        return y_hat * datamodule.target_std + datamodule.target_mean
