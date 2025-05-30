# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for semantic segmentation."""

import os
from typing import Any, Literal

import kornia.augmentation as K
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Accuracy, JaccardIndex, MetricCollection
from torchmetrics.classification import (
    Accuracy,
    FBetaScore,
    JaccardIndex,
    Precision,
    Recall,
)
from torchmetrics.wrappers import ClasswiseWrapper
from torchvision.models._api import WeightsEnum

from ..datasets import RGBBandsMissingError, unbind_samples
from ..models import FCN, get_weight
from . import utils
from .base import BaseTask


class SemanticSegmentationTask(BaseTask):
    """Semantic Segmentation."""

    def __init__(
        self,
        model: Literal['unet', 'deeplabv3+', 'fcn'] = 'unet',
        backbone: str = 'resnet50',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        task: Literal['binary', 'multiclass', 'multilabel'] = 'multiclass',
        num_classes: int | None = None,
        num_labels: int | None = None,
        labels: list[str] | None = None,
        num_filters: int = 3,
        loss: Literal['ce', 'bce', 'jaccard', 'focal'] = 'ce',
        class_weights: Tensor | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """Initialize a new SemanticSegmentationTask instance.

        Args:
            model: Name of the
                `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
            backbone: Name of the `timm
                <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or `smp
                <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False or
                None for random weights, or the path to a saved model state dict. FCN
                model does not support pretrained weights.
            in_channels: Number of input channels to model.
            task: One of 'binary', 'multiclass', or 'multilabel'.
            num_classes: Number of prediction classes (only for ``task='multiclass'``).
            num_labels: Number of prediction labels (only for ``task='multilabel'``).
            labels: Optional list of class label names. Must have length equal to
                ``num_classes`` when provided. If None, classwise metrics will use
                default integer names (e.g., "0", "1", "2"). Only used for
                ``task='multiclass'``.
            num_filters: Number of filters. Only applicable when model='fcn'.
            loss: Name of the loss function, currently supports
                'ce', 'bce', 'jaccard', and 'focal' loss.
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss.
            ignore_index: Optional integer class index to ignore in the loss and
                metrics.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.
            freeze_decoder: Freeze the decoder network to linear probe
                the segmentation head.

        .. versionadded:: 0.8
            The *labels* parameter.

        .. versionadded:: 0.7
           The *task* and *num_labels* parameters.

        .. versionchanged:: 0.6
           The *ignore_index* parameter now works for jaccard loss.

        .. versionadded:: 0.5
           The *class_weights*, *freeze_backbone*, and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           The *weights* parameter now supports WeightEnums and checkpoint paths.
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.

        .. versionchanged:: 0.4
           *segmentation_model*, *encoder_name*, and *encoder_weights*
           were renamed to *model*, *backbone*, and *weights*.

        .. versionchanged:: 0.3
           *ignore_zeros* was renamed to *ignore_index*.
        """
        self.weights = weights
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model."""
        model: str = self.hparams['model']
        backbone: str = self.hparams['backbone']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']
        num_classes: int = (
            self.hparams['num_classes'] or self.hparams['num_labels'] or 1
        )
        num_filters: int = self.hparams['num_filters']

        match model:
            case 'unet':
                self.model = smp.Unet(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels,
                    classes=num_classes,
                )
            case 'deeplabv3+':
                self.model = smp.DeepLabV3Plus(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels,
                    classes=num_classes,
                )
            case 'fcn':
                self.model = FCN(
                    in_channels=in_channels,
                    classes=num_classes,
                    num_filters=num_filters,
                )

        if model != 'fcn':
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams['freeze_backbone'] and model in ['unet', 'deeplabv3+']:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams['freeze_decoder'] and model in ['unet', 'deeplabv3+']:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        ignore_index: int | None = self.hparams['ignore_index']
        match self.hparams['loss']:
            case 'ce':
                ignore_value = -1000 if ignore_index is None else ignore_index
                self.criterion: nn.Module = nn.CrossEntropyLoss(
                    ignore_index=ignore_value, weight=self.hparams['class_weights']
                )
            case 'bce':
                self.criterion = nn.BCEWithLogitsLoss()
            case 'jaccard':
                # JaccardLoss requires a list of classes to use instead of a class
                # index to ignore.
                classes = [
                    i for i in range(self.hparams['num_classes']) if i != ignore_index
                ]

                self.criterion = smp.losses.JaccardLoss(
                    mode=self.hparams['task'], classes=classes
                )
            case 'focal':
                self.criterion = smp.losses.FocalLoss(
                    mode=self.hparams['task'],
                    ignore_index=ignore_index,
                    normalized=True,
                )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.classification.MulticlassAccuracy`: Overall accuracy
        (OA) using 'micro' averaging. The number of true positives divided by the
        dataset size. Higher values are better.
        * :class:`~torchmetrics.classification.MulticlassJaccardIndex`: Intersection
        over union (IoU). Uses 'micro' averaging. Higher valuers are better.

        .. note::
        * 'Micro' averaging suits overall performance evaluation but may not reflect
            minority class accuracy.
        * 'Macro' averaging gives equal weight to each class, useful
            for balanced performance assessment across imbalanced classes.
        * When providing class labels, ensure ``len(labels) == num_classes``. If
            labels are not provided, classwise metrics will use default integer names.
        """
        task: str = self.hparams['task']
        num_classes: int | None = self.hparams['num_classes']
        num_labels: int | None = self.hparams['num_labels']
        ignore_index: int | None = self.hparams['ignore_index']
        labels: list[str] | None = self.hparams.get('labels', None)
        
        # Validate labels length if provided
        if labels is not None and task in ['binary', 'multiclass']:
            expected_classes = 2 if task == 'binary' else num_classes
            if len(labels) != expected_classes:
                raise ValueError(
                    f"Length of labels ({len(labels)}) must equal number of classes "
                    f"({expected_classes}). Either provide {expected_classes} labels or "
                    f"set labels=None to use default integer names."
                )

        metrics_dict = {}

        # Common parameters for all metrics
        kwargs = {
            'task': task,
            'num_classes': num_classes,
            'num_labels': num_labels,
            'ignore_index': ignore_index,
        }

        # For binary and multilabel tasks, we only need micro averaging
        if task in ['binary', 'multilabel']:
            metrics_dict['OverallAccuracy'] = Accuracy(
                multidim_average='global',
                average='micro',
                **kwargs
            )
            metrics_dict['OverallF1Score'] = FBetaScore(
                multidim_average='global',
                average='micro',
                beta=1.0,
                **kwargs
            )
            metrics_dict['OverallJaccardIndex'] = JaccardIndex(
                average='micro',
                **kwargs
            )
            metrics_dict['OverallPrecision'] = Precision(
                multidim_average='global',
                average='micro',
                **kwargs
            )
            metrics_dict['OverallRecall'] = Recall(
                multidim_average='global',
                average='micro',
                **kwargs
            )
        else:  # multiclass task
            # Loop through averaging types for multiclass
            for average in ['micro', 'macro']:
                prefix = 'Overall' if average == 'micro' else 'Average'
                
                metrics_dict[f'{prefix}Accuracy'] = Accuracy(
                    multidim_average='global',
                    average=average,
                    **kwargs
                )
                
                metrics_dict[f'{prefix}F1Score'] = FBetaScore(
                    multidim_average='global',
                    average=average,
                    beta=1.0,
                    **kwargs
                )
                
                metrics_dict[f'{prefix}JaccardIndex'] = JaccardIndex(
                    average=average,
                    **kwargs
                )
                
                metrics_dict[f'{prefix}Precision'] = Precision(
                    multidim_average='global',
                    average=average,
                    **kwargs
                )
                
                metrics_dict[f'{prefix}Recall'] = Recall(
                    multidim_average='global',
                    average=average,
                    **kwargs
                )

        # Add classwise metrics for binary and multiclass (not multilabel)
        if task in ['binary', 'multiclass'] and labels is not None:
            metrics_dict['Accuracy'] = ClasswiseWrapper(
                Accuracy(
                    multidim_average='global',
                    average='none',
                    **kwargs
                ),
                labels=labels,
            )
            
            metrics_dict['F1Score'] = ClasswiseWrapper(
                FBetaScore(
                    multidim_average='global',
                    average='none',
                    beta=1.0,
                    **kwargs
                ),
                labels=labels,
            )
            
            metrics_dict['JaccardIndex'] = ClasswiseWrapper(
                JaccardIndex(
                    average='none',
                    **kwargs
                ),
                labels=labels,
            )
            
            metrics_dict['Precision'] = ClasswiseWrapper(
                Precision(
                    multidim_average='global',
                    average='none',
                    **kwargs
                ),
                labels=labels,
            )
            
            metrics_dict['Recall'] = ClasswiseWrapper(
                Recall(
                    multidim_average='global',
                    average='none',
                    **kwargs
                ),
                labels=labels,
            )

        # Create metric collections
        self.train_metrics = MetricCollection(metrics_dict, prefix='train_')
        self.val_metrics = self.train_metrics.clone(prefix='val_')
        self.test_metrics = self.train_metrics.clone(prefix='test_')

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
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics, batch_size=batch_size)

        if self.hparams['loss'] == 'bce':
            y = y.float()

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
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics, batch_size=batch_size)

        if self.hparams['loss'] == 'bce':
            y = y.float()

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
                case 'binary' | 'multilabel':
                    batch['prediction'] = (y_hat.sigmoid() >= 0.5).long()
                case 'multiclass':
                    batch['prediction'] = y_hat.argmax(dim=1)

            for key in ['image', 'mask', 'prediction']:
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
        y = batch['mask']
        batch_size = x.shape[0]
        y_hat = self(x).squeeze(1)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics, batch_size=batch_size)

        if self.hparams['loss'] == 'bce':
            y = y.float()

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
