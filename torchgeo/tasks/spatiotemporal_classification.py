# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for spatiotemporal classification at image level."""

from collections.abc import Sequence
from typing import Any, Literal

from torch import Tensor

from ..datasets.utils import Sample
from ..models import ConvLSTM
from .base import BaseTask
from .mixins import ClassificationMixin


class SpatioTemporalClassification(ClassificationMixin, BaseTask):
    """Classification for spatiotemporal inputs."""

    target_key = 'label'

    def __init__(
        self,
        model: Literal['convlstm'] | str = 'convlstm',
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
        """Initialize a new SpatioTemporalClassification instance.

        Args:
            model: Spatiotemporal model name. Supported value is ``'convlstm'``.
            in_channels: Number of channels per timestep for inputs of shape
                ``(B, T, C, H, W)``.
            task: Type of classification task, one of 'binary', 'multiclass', or
                'multilabel'.
            num_classes: Number of classes for classification.
            num_labels: Number of labels for multilabel classification.
            labels: Optional list of class names for computing metrics.
            pos_weight: Optional tensor of shape (num_classes,) for weighting the
                positive examples in binary/multilabel classification.
            loss: One of 'ce', 'bce', 'jaccard', 'focal', or 'dice'.
            class_weights: Optional tensor of class weights for handling imbalanced classes.
            ignore_index: Index of the class to ignore in the loss computation.
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
            **kwargs: Additional keyword arguments passed to the model.

        Returns:
            Output tensor of shape (B, num_outputs).
        """
        return self.model(x, **kwargs)

    def configure_models(self) -> None:
        """Initialize the model."""
        in_channels: int = self.hparams['in_channels']
        num_classes: int = (
            self.hparams['num_classes'] or self.hparams['num_labels'] or 1
        )
        self.model = ConvLSTM(
            input_dim=in_channels,
            num_classes=num_classes,
            convolutional_head=False,
            **self.kwargs,
        )

    def _shared_step(self, batch: Any, stage: str) -> Tensor:
        """Compute the loss and metrics for the given stage."""
        x = batch['image']
        y = batch[self.target_key]
        batch_size = x.shape[0]
        kwargs: dict[str, Tensor] = {}
        if (lengths := batch.get('length')) is not None:
            kwargs['lengths'] = lengths
        y_hat = self(x, **kwargs).squeeze(1)

        metrics = getattr(self, f'{stage}_metrics')
        metrics(y_hat, y)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss: Tensor = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss, batch_size=batch_size)
        return loss

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
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
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._shared_step(batch, 'val')

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        self._shared_step(batch, 'test')

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted class probabilities.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch['image']
        y_hat: Tensor = self(x)

        match self.hparams['task']:
            case 'binary' | 'multilabel':
                y_hat = y_hat.sigmoid()
            case 'multiclass':
                y_hat = y_hat.softmax(dim=1)

        return y_hat
