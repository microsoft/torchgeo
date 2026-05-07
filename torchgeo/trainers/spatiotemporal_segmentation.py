# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for spatiotemporal semantic segmentation."""

from collections.abc import Sequence
from typing import Any, Literal, cast

from torch import Tensor

from ..models import ConvLSTM
from .base import BaseTask
from .mixins import ClassificationMixin


class SpatioTemporalSegmentationTask(ClassificationMixin, BaseTask):
    """Spatiotemporal Semantic Segmentation.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        model: Literal['convlstm'] = 'convlstm',
        in_channels: int = 3,
        task: Literal['binary', 'multiclass', 'multilabel'] = 'multiclass',
        num_classes: int | None = None,
        num_labels: int | None = None,
        labels: list[str] | None = None,
        pos_weight: Tensor | None = None,
        loss: Literal['ce', 'bce', 'jaccard', 'focal', 'dice'] = 'ce',
        class_weights: Tensor | Sequence[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        **kwargs: Any,
    ) -> None:
        """Initialize a new SpatioTemporalSegmentationTask instance.

        Args:
            model: Spatiotemporal model name. Supported value is ``'convlstm'``.
            in_channels: Number of channels per timestep for inputs of shape
                ``(B, T, C, H, W)``.
            task: One of 'binary', 'multiclass', or 'multilabel'.
            num_classes: Number of prediction classes (only for ``task='multiclass'``).
            num_labels: Number of prediction labels (only for ``task='multilabel'``).
            labels: List of class names.
            pos_weight: A weight of positive examples and used with 'bce' loss.
            loss: Name of the loss function, currently supports
                'ce', 'bce', 'jaccard', 'focal', and 'dice' loss.
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss.
            ignore_index: Optional integer class index to ignore in the loss and
                metrics.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            **kwargs: Additional model-specific kwargs. For ``model='convlstm'``,
                kwargs are passed to ``ConvLSTM``.

        """
        super().__init__()

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, T, C, H, W).
            lengths: Optional sequence lengths (B,) before padding/truncation.

        Returns:
            Output tensor of shape (B, num_classes, H, W).
        """
        return self.model(x, lengths=lengths)

    def configure_models(self) -> None:
        """Initialize the model."""
        model: str = self.hparams['model']
        in_channels: int = self.hparams['in_channels']
        num_classes: int = (
            self.hparams['num_classes'] or self.hparams['num_labels'] or 1
        )

        match model:
            case 'convlstm':
                hidden_dim = cast(int | list[int], self.hparams.get('hidden_dim', 64))
                kernel_size = cast(
                    int | tuple[int, int] | list[int | tuple[int, int]],
                    self.hparams.get('kernel_size', 3),
                )
                num_layers = cast(int, self.hparams.get('num_layers', 1))
                head_kernel_size = cast(int, self.hparams.get('head_kernel_size', 1))
                bias = cast(bool, self.hparams.get('bias', True))
                return_all_layers = cast(
                    bool, self.hparams.get('return_all_layers', False)
                )
                self.model = ConvLSTM(
                    input_dim=in_channels,
                    hidden_dim=hidden_dim,
                    kernel_size=kernel_size,
                    num_layers=num_layers,
                    bias=bias,
                    return_all_layers=return_all_layers,
                    num_classes=num_classes,
                    head_kernel_size=head_kernel_size,
                )

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        """Compute the loss and metrics for the given stage."""
        x = batch['image']
        y = batch['mask']
        lengths = batch.get('length')
        batch_size = x.shape[0]
        y_hat = self(x, lengths=lengths).squeeze(1)

        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat, y)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss: Tensor = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss, batch_size=batch_size)
        return loss

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics."""
        del batch_idx, dataloader_idx
        return self._shared_step(batch, 'train')

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics."""
        del batch_idx, dataloader_idx
        self._shared_step(batch, 'val')

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics."""
        del batch_idx, dataloader_idx
        self._shared_step(batch, 'test')

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted class probabilities."""
        del batch_idx, dataloader_idx
        y_hat: Tensor = self(batch['image'], lengths=batch.get('length'))

        match self.hparams['task']:
            case 'binary' | 'multilabel':
                y_hat = y_hat.sigmoid()
            case 'multiclass':
                y_hat = y_hat.softmax(dim=1)

        return y_hat
