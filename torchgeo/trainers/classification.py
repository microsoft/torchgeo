# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for image classification."""

import os
from typing import Any, Literal

import kornia.augmentation as K
import matplotlib.pyplot as plt
import timm
import torch.nn as nn
from matplotlib.figure import Figure
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, FBetaScore, JaccardIndex
from torchvision.models._api import WeightsEnum
from typing_extensions import deprecated

from ..datasets import RGBBandsMissingError, unbind_samples
from ..models import get_weight
from . import utils
from .base import BaseTask


class ClassificationTask(BaseTask):
    """Image classification."""

    def __init__(
        self,
        model: str = 'resnet50',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        task: Literal['binary', 'multiclass', 'multilabel'] = 'multiclass',
        num_classes: int | None = None,
        num_labels: int | None = None,
        loss: Literal['ce', 'bce', 'jaccard', 'focal'] = 'ce',
        class_weights: Tensor | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize a new ClassificationTask instance.

        Args:
            model: Name of the `timm
                <https://huggingface.co/docs/timm/reference/models>`__ model to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            task: One of 'binary', 'multiclass', or 'multilabel'.
            num_classes: Number of prediction classes (only for ``task='multiclass'``).
            num_labels: Number of prediction labels (only for ``task='multilabel'``).
            loss: One of 'ce', 'bce', 'jaccard', or 'focal'.
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to linear probe
                the classifier head.

        .. versionadded:: 0.7
           The *task* and *num_labels* parameters.

        .. versionadded:: 0.5
           The *class_weights* and *freeze_backbone* parameters.

        .. versionchanged:: 0.5
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.

        .. versionchanged:: 0.4
           *classification_model* was renamed to *model*.
        """
        self.weights = weights
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights

        # Create model
        self.model = timm.create_model(
            self.hparams['model'],
            num_classes=self.hparams['num_classes'] or self.hparams['num_labels'] or 1,
            in_chans=self.hparams['in_channels'],
            pretrained=weights is True,
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hparams['freeze_backbone']:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        match self.hparams['loss']:
            case 'ce':
                self.criterion: nn.Module = nn.CrossEntropyLoss(
                    weight=self.hparams['class_weights']
                )
            case 'bce':
                self.criterion = nn.BCEWithLogitsLoss()
            case 'jaccard':
                self.criterion = JaccardLoss(mode=self.hparams['task'])
            case 'focal':
                self.criterion = FocalLoss(mode=self.hparams['task'], normalized=True)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.Accuracy`: The number of
          true positives divided by the dataset size. Both overall accuracy (OA)
          using 'micro' averaging and average accuracy (AA) using 'macro' averaging
          are reported. Higher values are better.
        * :class:`~torchmetrics.JaccardIndex`: Intersection
          over union (IoU). Uses 'macro' averaging. Higher valuers are better.
        * :class:`~torchmetrics.FBetaScore`: F1 score.
          The harmonic mean of precision and recall. Uses 'micro' averaging.
          Higher values are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging gives equal weight to each class, and is useful for
             balanced performance assessment across imbalanced classes.
        """
        kwargs = {
            'task': self.hparams['task'],
            'num_classes': self.hparams['num_classes'],
            'num_labels': self.hparams['num_labels'],
        }
        metrics = MetricCollection(
            {
                'OverallAccuracy': Accuracy(average='micro', **kwargs),
                'AverageAccuracy': Accuracy(average='macro', **kwargs),
                'JaccardIndex': JaccardIndex(**kwargs),
                'F1Score': FBetaScore(beta=1.0, average='micro', **kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

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
        x = batch['image']
        y = batch['label']
        batch_size = x.shape[0]
        y_hat = self(x)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size)

        if self.hparams['loss'] == 'bce':
            y = y.float()
            y_hat = y_hat.squeeze(1)

        loss: Tensor = self.criterion(y_hat, y)
        self.log('train_loss', loss, batch_size=batch_size)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
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
        y_hat = self(x)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size)

        if self.hparams['loss'] == 'bce':
            y = y.float()
            y_hat = y_hat.squeeze(1)

        loss = self.criterion(y_hat, y)
        self.log('val_loss', loss, batch_size=batch_size)

        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
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
                case 'binary':
                    batch['prediction'] = (y_hat >= 0.5).long()
                case 'multiclass':
                    batch['prediction'] = y_hat.argmax(dim=1)
                case 'multilabel':
                    batch['prediction'] = (y_hat >= 0.5).long()

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
                )
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = batch['label']
        batch_size = x.shape[0]
        y_hat = self(x)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        if self.hparams['loss'] == 'bce':
            y = y.float()
            y_hat = y_hat.squeeze(1)

        loss = self.criterion(y_hat, y)
        self.log('test_loss', loss, batch_size=batch_size)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
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


@deprecated('Use torchgeo.trainers.ClassificationTask instead')
class MultiLabelClassificationTask(ClassificationTask):
    """Multi-label image classification."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Wrapper around torchgeo.trainers.ClassificationTask to massage kwargs."""
        kwargs['task'] = 'multilabel'
        kwargs['num_labels'] = kwargs['num_classes']
        super().__init__(*args, **kwargs)
