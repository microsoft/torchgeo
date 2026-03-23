# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for video regression."""

from typing import Literal, cast

import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from ..datasets.utils import Sample
from ..models import ConvLSTM
from .base import BaseTask


class _ConvLSTMRegression(nn.Module):
    """ConvLSTM backbone with a regression head."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int = 1,
        hidden_dim: int | list[int] = 64,
        kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        num_layers: int = 1,
        head_kernel_size: int = 1,
    ) -> None:
        super().__init__()
        self.backbone = ConvLSTM(
            input_dim=in_channels,
            hidden_dim=hidden_dim,
            kernel_size=kernel_size,
            num_layers=num_layers,
            batch_first=True,
            return_all_layers=False,
        )
        head_in_channels = (
            hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
        )
        padding = head_kernel_size // 2
        self.head = nn.Conv2d(
            in_channels=head_in_channels,
            out_channels=num_outputs,
            kernel_size=head_kernel_size,
            padding=padding,
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, T, C, H, W).
            lengths: Optional sequence lengths (B,) before padding/truncation.
                Values larger than the available sequence length use the final
                timestep.

        Returns:
            Output tensor of shape (B, num_outputs, H, W).
        """
        layer_output_list, _ = self.backbone(x)
        layer_output = layer_output_list[-1]

        if lengths is None:
            features = layer_output[:, -1]
        else:
            idx = lengths.to(device=layer_output.device, dtype=torch.long) - 1
            idx = idx.clamp(min=0, max=layer_output.size(1) - 1)
            batch_idx = torch.arange(layer_output.size(0), device=idx.device)
            features = layer_output[batch_idx, idx]

        return cast(Tensor, self.head(features))


class VideoPixelwiseRegressionTask(BaseTask):
    """Pixelwise regression for video inputs."""

    target_key = 'mask'

    def __init__(
        self,
        model: Literal['convlstm'] | str = 'convlstm',
        in_channels: int = 3,
        num_outputs: int = 1,
        loss: str = 'mse',
        lr: float = 1e-3,
        patience: int = 10,
        convlstm_hidden_dim: int | list[int] = 64,
        convlstm_kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        convlstm_num_layers: int = 1,
        convlstm_head_kernel_size: int = 1,
    ) -> None:
        """Initialize a new VideoPixelwiseRegressionTask instance.

        Args:
            model: Video model name. Only ``'convlstm'`` is currently supported.
                The explicit model switch is kept so more video architectures can
                be added later without reshaping the trainer API.
            in_channels: Number of channels per timestep for inputs of shape
                ``(B, T, C, H, W)``.
            num_outputs: Number of prediction outputs per pixel.
            loss: One of 'mse' or 'mae'.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            convlstm_hidden_dim: Hidden dimension(s) for ``model='convlstm'``.
            convlstm_kernel_size: Kernel size(s) for ``model='convlstm'``.
            convlstm_num_layers: Number of layers for ``model='convlstm'``.
            convlstm_head_kernel_size: Kernel size for the conv regression head
                when using ``model='convlstm'``.
        """
        super().__init__()

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, T, C, H, W).
            lengths: Optional sequence lengths (B,) before padding/truncation.

        Returns:
            Output tensor of shape (B, num_outputs, H, W).
        """
        return self.model(x, lengths=lengths)

    def configure_models(self) -> None:
        """Initialize the model."""
        model: str = self.hparams['model']
        in_channels: int = self.hparams['in_channels']
        num_outputs: int = self.hparams['num_outputs']
        convlstm_hidden_dim: int | list[int] = self.hparams['convlstm_hidden_dim']
        convlstm_kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = (
            self.hparams['convlstm_kernel_size']
        )
        convlstm_num_layers: int = self.hparams['convlstm_num_layers']
        convlstm_head_kernel_size: int = self.hparams['convlstm_head_kernel_size']

        match model:
            case 'convlstm':
                self.model = _ConvLSTMRegression(
                    in_channels=in_channels,
                    num_outputs=num_outputs,
                    hidden_dim=convlstm_hidden_dim,
                    kernel_size=convlstm_kernel_size,
                    num_layers=convlstm_num_layers,
                    head_kernel_size=convlstm_head_kernel_size,
                )
            case _:
                raise ValueError(
                    f"Model type '{model}' is not supported. "
                    'Currently, VideoPixelwiseRegressionTask only supports '
                    "'convlstm'."
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

    def _shared_step(self, batch: Sample, stage: str) -> Tensor:
        """Compute the loss and metrics for the given stage."""
        x = batch['image']
        y = batch[self.target_key].to(torch.float)
        lengths = batch.get('length')
        batch_size = x.shape[0]
        y_hat = self(x, lengths=lengths)

        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)

        loss: Tensor = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss, batch_size=batch_size)

        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat, y)
        self.log_dict(metrics, batch_size=batch_size)

        return loss

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics."""
        del batch_idx, dataloader_idx
        return self._shared_step(batch, 'train')

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics."""
        del batch_idx, dataloader_idx
        self._shared_step(batch, 'val')

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics."""
        del batch_idx, dataloader_idx
        self._shared_step(batch, 'test')

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted regression values."""
        del batch_idx, dataloader_idx
        return self(batch['image'], lengths=batch.get('length'))
