# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for spatiotemporal pixelwise regression."""

from typing import Any, Literal

import torch
from torch import Tensor

from ..models import ConvLSTM
from .base import BaseTask
from .mixins import RegressionMixin


class SpatioTemporalPixelwiseRegressionTask(RegressionMixin, BaseTask):
    """LightningModule for pixelwise regression over spatiotemporal image sequences.

    Uses a :class:`~torchgeo.models.ConvLSTM` encoder with a single-channel
    regression head to predict a continuous per-pixel value from a sequence of
    images of shape ``(B, T, C, H, W)``.

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
        """Initialize a new SpatioTemporalPixelwiseRegressionTask instance.

        Args:
            model: Spatiotemporal model name. Currently only ``'convlstm'`` is
                supported.
            in_channels: Number of channels per timestep for inputs of shape
                ``(B, T, C, H, W)``.
            num_outputs: Number of output channels. Defaults to ``1`` for
                single-channel pixelwise regression.
            labels: List of output channel names used when ``num_outputs > 1``.
            loss: Loss function, one of ``'mse'`` or ``'mae'``.
            lr: Learning rate for the optimizer.
            patience: Patience for the learning-rate scheduler.
            **kwargs: Additional keyword arguments forwarded to the
                :class:`~torchgeo.models.ConvLSTM` constructor, e.g.
                ``hidden_dim``, ``num_layers``, ``kernel_size``, and
                ``head_kernel_size``.
        """
        self.kwargs = kwargs
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the ConvLSTM model with a single-channel regression head."""
        self.model = ConvLSTM(
            input_dim=self.hparams['in_channels'], num_classes=1, **self.kwargs
        )

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape ``(B, T, C, H, W)``.
            **kwargs: Additional keyword arguments forwarded to the model,
                e.g. ``lengths`` for variable-length sequences.

        Returns:
            Output tensor of shape ``(B, 1, H, W)``.
        """
        return self.model(x, **kwargs)

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        """Compute the loss and metrics for a given stage.

        Args:
            batch: The output of your DataLoader.  Must contain ``'image'``
                (shape ``(B, T, C, H, W)``) and ``'mask'``
                (shape ``(B, H, W)``, float).  May optionally contain
                ``'length'`` (shape ``(B,)``) for variable-length sequences.
            stage: One of ``'train'``, ``'val'``, or ``'test'``.

        Returns:
            The scalar loss tensor.
        """
        x = batch['image']
        y = batch['mask'].to(torch.float)
        batch_size = x.shape[0]

        kwargs: dict[str, Tensor] = {}
        if (lengths := batch.get('length')) is not None:
            kwargs['lengths'] = lengths

        y_hat = self(x, **kwargs).squeeze(1)  # (B, H, W)

        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat, y)

        loss: Tensor = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss, batch_size=batch_size)
        return loss

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        return self._shared_step(batch, 'train')

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._shared_step(batch, 'val')

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._shared_step(batch, 'test')

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted regression values.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output tensor of shape ``(B, 1, H, W)``.
        """
        kwargs: dict[str, Tensor] = {}
        if (lengths := batch.get('length')) is not None:
            kwargs['lengths'] = lengths
        y_hat: Tensor = self(batch['image'], **kwargs)
        return y_hat
