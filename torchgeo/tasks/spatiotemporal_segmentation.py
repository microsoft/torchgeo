# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for spatiotemporal semantic segmentation."""

from collections.abc import Sequence
from typing import Any, Literal, cast

from torch import Tensor

from ..models import UTAE, ConvLSTM
from .base import BaseTask
from .mixins import ClassificationMixin


class SpatioTemporalSegmentation(ClassificationMixin, BaseTask):
    """Spatiotemporal Semantic Segmentation.

    .. versionadded:: 0.10
    """

    def __init__(
        self,
        model: Literal['convlstm', 'utae'] = 'convlstm',
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
        """Initialize a new SpatioTemporalSegmentation instance.

        Args:
            model: Spatiotemporal model name. One of ``'convlstm'`` or
                ``'utae'``.
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
            **kwargs: Additional keyword arguments passed to the model constructor.
        """
        self.kwargs = kwargs
        super().__init__()

    def forward(self, x: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape (B, T, C, H, W).
            **kwargs: Additional keyword arguments forwarded to the model.

        Returns:
            Output tensor of shape (B, num_classes, H, W).
        """
        return self.model(x, **kwargs)

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If ``model`` is invalid.
        """
        in_channels: int = self.hparams['in_channels']
        num_classes: int = (
            self.hparams['num_classes'] or self.hparams['num_labels'] or 1
        )
        match self.hparams['model']:
            case 'convlstm':
                self.model = ConvLSTM(
                    input_dim=in_channels, num_classes=num_classes, **self.kwargs
                )
            case 'utae':
                kwargs = dict(self.kwargs)
                out_conv = cast(
                    Sequence[int], kwargs.pop('out_conv', (32, num_classes))
                )
                self.model = UTAE(input_dim=in_channels, out_conv=out_conv, **kwargs)
            case _:
                model = self.hparams['model']
                raise ValueError(f"Model type '{model}' is not valid.")

    def _model_kwargs(self, batch: Any) -> dict[str, Tensor]:
        """Extract model-specific keyword arguments from a batch."""
        kwargs: dict[str, Tensor] = {}
        match self.hparams['model']:
            case 'convlstm':
                if (lengths := batch.get('length')) is not None:
                    kwargs['lengths'] = lengths
            case 'utae':
                if (batch_positions := batch.get('batch_positions')) is not None:
                    kwargs['batch_positions'] = batch_positions
        return kwargs

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        """Compute the loss and metrics for the given stage."""
        x = batch['image']
        y = batch['mask']
        batch_size = x.shape[0]
        kwargs = self._model_kwargs(batch)
        y_hat = self(x, **kwargs).squeeze(1)

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
        return self._shared_step(batch, 'train')

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics."""
        self._shared_step(batch, 'val')

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics."""
        self._shared_step(batch, 'test')

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted class probabilities."""
        kwargs = self._model_kwargs(batch)
        y_hat: Tensor = self(batch['image'], **kwargs)

        match self.hparams['task']:
            case 'binary' | 'multilabel':
                y_hat = y_hat.sigmoid()
            case 'multiclass':
                y_hat = y_hat.softmax(dim=1)

        return y_hat
