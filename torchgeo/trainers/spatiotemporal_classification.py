# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for spatiotemporal regression."""

from collections.abc import Sequence
from typing import Literal

import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor

from ..datamodules import BaseDataModule
from ..datasets import RGBBandsMissingError, unbind_samples
from ..datasets.utils import Sample
from ..models import ConvLSTM
from .base import BaseTask
from .mixins import ClassificationMixin


class _ConvLSTMClassifier(nn.Module):
    """ConvLSTM backbone with a classification head."""

    def __init__(
        self,
        in_channels: int,
        num_outputs: int = 1,
        hidden_dim: int | list[int] = 64,
        kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        num_layers: int = 1,
        dropout: float = 0.0,
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
        feat_dim = hidden_dim[-1] if isinstance(hidden_dim, list) else hidden_dim
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(feat_dim, num_outputs)
        )

    def forward(self, x: Tensor, lengths: Tensor | None = None) -> Tensor:
        layer_output_list, _ = self.backbone(x)
        layer_output = layer_output_list[-1]

        if lengths is None:
            features = layer_output[:, -1]
        else:
            idx = lengths.to(device=layer_output.device, dtype=torch.long) - 1
            idx = idx.clamp(min=0, max=layer_output.size(1) - 1)
            batch_idx = torch.arange(layer_output.size(0), device=idx.device)
            features = layer_output[batch_idx, idx]

        pooled = self.pool(features)  # (B, C, 1, 1)
        logits = self.classifier(pooled)  # (B, num_outputs)
        return logits


class SpatioTemporalClassificationTask(ClassificationMixin, BaseTask):
    """Classification for spatiotemporal inputs."""

    target_key = 'mask'

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
        convlstm_hidden_dim: int | list[int] = 64,
        convlstm_kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = 3,
        convlstm_num_layers: int = 1,
        convlstm_dropout: float = 0.0,
    ) -> None:
        """Initialize a new SpatioTemporalPixelwiseClassificationTask instance.

        Args:
            model: Video model name. Only ``'convlstm'`` is currently supported.
                The explicit model switch is kept so more video architectures can
                be added later without reshaping the trainer API.
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
            convlstm_hidden_dim: Hidden dimension(s) for ``model='convlstm'``.
            convlstm_kernel_size: Kernel size(s) for ``model='convlstm'``.
            convlstm_num_layers: Number of layers for ``model='convlstm'``.
            convlstm_dropout: Dropout probability for ``model='convlstm'``.
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
        num_classes: int = self.hparams['num_classes']
        convlstm_hidden_dim: int | list[int] = self.hparams['convlstm_hidden_dim']
        convlstm_kernel_size: int | tuple[int, int] | list[int | tuple[int, int]] = (
            self.hparams['convlstm_kernel_size']
        )
        convlstm_num_layers: int = self.hparams['convlstm_num_layers']
        convlstm_dropout: float = self.hparams['convlstm_dropout']
        match model:
            case 'convlstm':
                self.model = _ConvLSTMClassifier(
                    in_channels=in_channels,
                    num_outputs=num_classes,
                    hidden_dim=convlstm_hidden_dim,
                    kernel_size=convlstm_kernel_size,
                    num_layers=convlstm_num_layers,
                    dropout=convlstm_dropout,
                )
            case _:
                raise ValueError(
                    f"Model type '{model}' is not supported. "
                    'Currently, SpatioTemporalPixelwiseClassificationTask only supports '
                    "'convlstm'."
                )

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
        x = batch['image']
        y = batch['label']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.train_metrics(y_hat, y)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss: Tensor = self.criterion(y_hat, y)
        self.log('train_loss', loss, batch_size=batch_size)

        return loss

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['label']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.val_metrics(y_hat, y)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size)

        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and isinstance(self.trainer.datamodule, BaseDataModule)
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            datamodule = self.trainer.datamodule
            aug = K.AugmentationSequential(
                K.Denormalize(datamodule.mean, datamodule.std),
                data_keys=None,
                keepdim=True,
            )
            batch = aug(batch)
            match self.hparams['task']:
                case 'binary' | 'multilabel':
                    batch['prediction'] = (y_hat.sigmoid() >= 0.5).long()
                case 'multiclass':
                    batch['prediction'] = y_hat.argmax(dim=1)

            for key in ['image', 'label', 'prediction']:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Figure | None = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f'image/{batch_idx}', fig, global_step=self.global_step
                )  # ty: ignore[call-non-callable]
                plt.close()

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['label']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.test_metrics(y_hat, y)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, batch_size=batch_size)

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
