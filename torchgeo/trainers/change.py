# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for change detection."""

import os
from typing import Any, Literal

import kornia.augmentation as K
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from einops import rearrange
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryF1Score,
    BinaryJaccardIndex,
)
from torchvision.models._api import WeightsEnum

from ..datasets import RGBBandsMissingError, unbind_samples
from ..models import FCN, FCSiamConc, FCSiamDiff, get_weight
from . import utils
from .base import BaseTask


class ChangeDetectionTask(BaseTask):
    """Change Detection. Currently supports binary change between two timesteps.

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
        ] = 'unet',
        backbone: str = 'resnet50',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        pos_weight: Tensor | None = None,
        loss: Literal['bce', 'jaccard', 'focal'] = 'bce',
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
        num_filters: int = 3,
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
            pos_weight: A weight of positive examples and used with 'bce' loss.
            loss: Name of the loss function, currently supports
                'bce', 'jaccard', or 'focal' loss.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.
            freeze_decoder: Freeze the decoder network to linear probe
                the segmentation head.
            num_filters: Number of filters. Only applicable when model='fcn'.
        """
        self.weights = weights
        super().__init__()

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        match self.hparams['loss']:
            case 'bce':
                self.criterion = nn.BCEWithLogitsLoss(
                    pos_weight=self.hparams['pos_weight']
                )
            case 'jaccard':
                self.criterion = smp.losses.JaccardLoss(mode='binary')
            case 'focal':
                self.criterion = smp.losses.FocalLoss(mode='binary', normalized=True)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        metrics = MetricCollection(
            {
                'accuracy': BinaryAccuracy(),
                'jaccard': BinaryJaccardIndex(),
                'f1': BinaryF1Score(),
            }
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
        num_classes = 1

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
                    classes=1,
                )
            case 'fcn':
                self.model = FCN(
                    in_channels=in_channels * 2,  # images are concatenated
                    classes=num_classes,
                    num_filters=self.hparams['num_filters'],
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

        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams['freeze_backbone'] and model != 'fcn':
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams['freeze_decoder'] and model != 'fcn':
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
        # channel dim for binary loss functions/metrics
        y = rearrange(y, 'b h w -> b () h w')
        if model == 'unet':
            x = rearrange(x, 'b t c h w -> b (t c) h w')
        y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y.to(torch.float))
        self.log(f'{stage}_loss', loss)

        # Retrieve the correct metrics based on the stage
        metrics = getattr(self, f'{stage}_metrics', None)
        if metrics:
            metrics(y_hat, y)
            self.log_dict({f'{k}': v for k, v in metrics.compute().items()})

        if stage in ['val']:
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
                batch['prediction'] = (y_hat.sigmoid() >= 0.5).long()
                # Remove channel dim from mask
                batch['prediction'] = rearrange(
                    batch['prediction'], 'b () h w -> b h w'
                )
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
        y_hat: Tensor = self(x)
        y_hat = y_hat.sigmoid()
        return y_hat
