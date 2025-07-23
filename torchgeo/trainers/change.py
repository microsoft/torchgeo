# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for change detection."""

import os
from collections.abc import Sequence
from typing import Any, Literal

import kornia.augmentation as K
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from einops import rearrange
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import Accuracy, F1Score, JaccardIndex, MetricCollection
from torchvision.models._api import WeightsEnum

from ..datasets import RGBBandsMissingError, unbind_samples
from ..models import (
    FCN,
    FCSiamConc,
    FCSiamDiff,
    changevit_small,
    changevit_tiny,
    get_weight,
)
from . import utils
from .base import BaseTask


class ChangeDetectionTask(BaseTask):
    """Change Detection. Supports binary, multiclass, and multilabel change detection.

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        model: Literal[
            'unet',
            'deeplabv3+',
            'fcn',
            'upernet',
            'segformer',
            'dpt',
            'fcsiamdiff',
            'fcsiamconc',
            'changevit_small',
            'changevit_tiny',
        ] = 'unet',
        backbone: str = 'resnet50',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        task: Literal['binary', 'multiclass', 'multilabel'] = 'binary',
        num_classes: int | None = None,
        num_labels: int | None = None,
        num_filters: int = 3,
        pos_weight: Tensor | None = None,
        loss: Literal['ce', 'bce', 'jaccard', 'focal'] = 'bce',
        class_weights: Tensor | Sequence[float] | None = None,
        ignore_index: int | None = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """Initialize a new ChangeDetectionTask instance.

        Args:
            model: Name of the model to use.
            backbone: Name of the `timm
                <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or `smp
                <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False or
                None for random weights, or the path to a saved model state dict. FCN
                model does not support pretrained weights.
            in_channels: Number of channels per image.
            task: One of 'binary', 'multiclass', or 'multilabel'.
            num_classes: Number of prediction classes (only for ``task='multiclass'``).
            num_labels: Number of prediction labels (only for ``task='multilabel'``).
            num_filters: Number of filters. Only applicable when model='fcn'.
            pos_weight: A weight of positive examples and used with 'bce' loss.
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
        """
        self.weights = weights
        super().__init__()

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        ignore_index: int | None = self.hparams['ignore_index']
        class_weights = self.hparams['class_weights']
        if class_weights is not None and not isinstance(class_weights, Tensor):
            class_weights = torch.tensor(class_weights, dtype=torch.float32)

        match self.hparams['loss']:
            case 'ce':
                ignore_value = -1000 if ignore_index is None else ignore_index
                self.criterion: nn.Module = nn.CrossEntropyLoss(
                    ignore_index=ignore_value, weight=class_weights
                )
            case 'bce':
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=self.hparams['pos_weight']
                )
            case 'jaccard':
                # JaccardLoss requires a list of classes to use instead of a class
                # index to ignore.
                if self.hparams['task'] == 'multiclass' and ignore_index is not None:
                    classes = [
                        i
                        for i in range(self.hparams['num_classes'])
                        if i != ignore_index
                    ]
                    self.criterion = smp.losses.JaccardLoss(
                        mode=self.hparams['task'], classes=classes
                    )
                else:
                    self.criterion = smp.losses.JaccardLoss(mode=self.hparams['task'])
            case 'focal':
                self.criterion = smp.losses.FocalLoss(
                    mode=self.hparams['task'],
                    ignore_index=ignore_index,
                    normalized=True,
                )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.Accuracy`: Overall accuracy
          (OA) using 'micro' averaging. The number of true positives divided by the
          dataset size. Higher values are better.
        * :class:`~torchmetrics.JaccardIndex`: Intersection
          over union (IoU). Uses 'micro' averaging. Higher valuers are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging, not used here, gives equal weight to each class, useful
             for balanced performance assessment across imbalanced classes.
        """
        kwargs = {
            'task': self.hparams['task'],
            'num_classes': self.hparams['num_classes'],
            'num_labels': self.hparams['num_labels'],
            'ignore_index': self.hparams['ignore_index'],
        }
        metrics = MetricCollection(
            [
                Accuracy(multidim_average='global', average='micro', **kwargs),
                JaccardIndex(average='micro', **kwargs),
                F1Score(average='micro', **kwargs),
            ]
        )
        self.train_metrics = metrics.clone(prefix='train_')
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

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
                    in_channels=in_channels * 2,  # images are concatenated
                    classes=num_classes,
                )
            case 'deeplabv3+':
                self.model = smp.DeepLabV3Plus(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels * 2,  # images are concatenated
                    classes=num_classes,
                )
            case 'fcn':
                self.model = FCN(
                    in_channels=in_channels * 2,  # images are concatenated
                    classes=num_classes,
                    num_filters=num_filters,
                )
            case 'upernet':
                self.model = smp.UPerNet(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels * 2,  # images are concatenated
                    classes=num_classes,
                )
            case 'segformer':
                self.model = smp.Segformer(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels * 2,  # images are concatenated
                    classes=num_classes,
                )
            case 'dpt':
                self.model = smp.DPT(
                    encoder_name=backbone,
                    encoder_weights='imagenet' if weights is True else None,
                    in_channels=in_channels * 2,  # images are concatenated
                    classes=num_classes,
                )
            case 'fcsiamdiff':
                self.model = FCSiamDiff(
                    encoder_name=backbone,
                    in_channels=in_channels,
                    classes=num_classes,
                    encoder_weights='imagenet' if weights is True else None,
                )
            case 'fcsiamconc':
                self.model = FCSiamConc(
                    encoder_name=backbone,
                    in_channels=in_channels,
                    classes=num_classes,
                    encoder_weights='imagenet' if weights is True else None,
                )
            case 'changevit_small':
                self.model = changevit_small(
                    weights=weights if isinstance(weights, WeightsEnum) else None
                )
            case 'changevit_tiny':
                self.model = changevit_tiny(
                    weights=weights if isinstance(weights, WeightsEnum) else None
                )

        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)

            # For ChangeViT models, only load backbone weights
            if model.startswith('changevit'):
                # Load ViT backbone weights only
                vit_state_dict = {
                    k.replace('vit_backbone.', ''): v
                    for k, v in state_dict.items()
                    if k.startswith('vit_backbone.')
                }
                if vit_state_dict:
                    self.model.vit_backbone.load_state_dict(
                        vit_state_dict, strict=False
                    )
            else:
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams['freeze_backbone'] and model != 'fcn':
            if model.startswith('changevit'):
                # Freeze ViT backbone for ChangeViT models
                for param in self.model.vit_backbone.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.encoder.parameters():
                    param.requires_grad = False

        # Freeze decoder
        if self.hparams['freeze_decoder'] and model != 'fcn':
            if model.startswith('changevit'):
                # Freeze detail capture and feature injector for ChangeViT models
                for param in self.model.detail_capture.parameters():
                    param.requires_grad = False
                for param in self.model.feature_injector.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.decoder.parameters():
                    param.requires_grad = False

    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        """Compute the loss and additional metrics for the given stage.

        Args:
            batch: The output of your DataLoader._
            batch_idx: Integer displaying index of this batch._
            stage: The current stage.

        Returns:
            The loss tensor.
        """
        model: str = self.hparams['model']
        x = batch['image']
        y = batch['mask']

        if not model.startswith('fcsiam') and not model.startswith('changevit'):
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        y_hat = self(x)

        if self.hparams['task'] == 'multiclass':
            y = y.squeeze(1)
        elif self.hparams['task'] == 'multiclass':
            y = y.squeeze(1)
            y = y.long()

        # Forward pass
        if model.startswith('changevit'):
            output = self(x)
            # ChangeViT outputs probabilities in both training and inference modes
            y_hat = output['change_prob']
        else:
            y_hat = self(x)

        if self.hparams['loss'] == 'bce':
            y = y.float()

        # Compute the loss
        loss: Tensor = self.criterion(y_hat, y)
        self.log(f'{stage}_loss', loss)

        # Retrieve the correct metrics based on the stage
        metrics = getattr(self, f'{stage}_metrics', None)
        if metrics:
            # Transform predictions for metrics calculation
            match self.hparams['task']:
                case 'binary' | 'multilabel':
                    y_hat = (y_hat.sigmoid() >= 0.5).long()
                case 'multiclass':
                    y_hat = y_hat.argmax(dim=1)

            metrics(y_hat, y)
            self.log_dict(metrics, batch_size=x.shape[0])

        if stage == 'val':
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
                    K.VideoSequential(K.Denormalize(datamodule.mean, datamodule.std)),
                    data_keys=None,
                    keepdim=True,
                )
                batch = aug(batch)
                match self.hparams['task']:
                    case 'binary' | 'multilabel':
                        prediction = (y_hat.sigmoid() >= 0.5).long()
                        # Restore channel dimension for plotting compatibility
                        batch['prediction'] = prediction.unsqueeze(1)
                    case 'multiclass':
                        batch['prediction'] = y_hat.argmax(dim=1, keepdim=True)

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

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.

        Returns:
            The loss tensor.
        """
        loss = self._shared_step(batch, batch_idx, 'train')
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        """Compute the validation loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
        """
        self._shared_step(batch, batch_idx, 'val')

    def test_step(self, batch: Any, batch_idx: int) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
        """
        self._shared_step(batch, batch_idx, 'test')

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
        model: str = self.hparams['model']
        x = batch['image']
        if model == 'unet':
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        elif model.startswith('changevit'):
            # ChangeViT expects bitemporal format [B, T, C, H, W]
            pass  # x is already in correct format
        else:
            # For other models that expect concatenated input
            if len(x.shape) == 5:  # [B, T, C, H, W]
                x = rearrange(x, 'b t c h w -> b (t c) h w')

        # Forward pass
        if model.startswith('changevit'):
            output = self(x)
            y_hat: Tensor = (
                output['change_prob']
                if 'change_prob' in output
                else output['bi_change_logit'].sigmoid()
            )
        else:
            y_hat = self(x)

        match self.hparams['task']:
            case 'binary' | 'multilabel':
                if not model.startswith(
                    'changevit'
                ):  # ChangeViT already applies sigmoid
                    y_hat = y_hat.sigmoid()
            case 'multiclass':
                y_hat = y_hat.softmax(dim=1)

        return y_hat
