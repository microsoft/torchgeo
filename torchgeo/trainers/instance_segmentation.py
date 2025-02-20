# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for instance segmentation."""

import os
from typing import Any

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models._api import WeightsEnum
from torchvision.models.detection import MaskRCNN

from ..datasets import RGBBandsMissingError, unbind_samples
from ..models import get_weight
from . import utils
from .base import BaseTask


class InstanceSegmentationTask(BaseTask):
    """Instance Segmentation.

    .. versionadded:: 0.7
    """

    def __init__(
        self,
        model: str = 'mask_rcnn',
        backbone: str = 'resnet50',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        num_classes: int = 2,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize a new InstanceSegmentationTask instance.

        Args:
            model: Name of the model to use.
            backbone: Name of the backbone to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False or
                None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes (including the background).
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.
        """
        self.weights = weights
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        # Create backbone
        backbone = timm.create_model(  # type: ignore[attr-defined]
            self.hparams['backbone'],
            in_chans=self.hparams['in_channels'],
            pretrained=self.weights is True,
        )

        # Compatibility with torchvision
        backbone.out_channels = backbone.num_classes

        # Freeze backbone
        if self.hparams['freeze_backbone']:
            for param in backbone.parameters():
                param.requires_grad = False

        # Load weights
        if self.weights and self.weights is not True:
            if isinstance(self.weights, WeightsEnum):
                state_dict = self.weights.get_state_dict(progress=True)
            elif os.path.exists(self.weights):
                _, state_dict = utils.extract_backbone(self.weights)
            else:
                state_dict = get_weight(self.weights).get_state_dict(progress=True)
            utils.load_state_dict(backbone, state_dict)

        # Create model
        match model := self.hparams['model']:
            case 'mask_rcnn':
                self.model = MaskRCNN(backbone, self.hparams['num_classes'])
                self.model.transform = nn.Sequential()  # no-op
            case _:
                msg = f"Invalid model type '{model}'. Supported model: 'mask_rcnn'"
                raise ValueError(msg)

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        - Uses Mean Average Precision (mAP) for masks (IOU-based metric).
        """
        metrics = MetricCollection([MeanAveragePrecision(iou_type='segm')])
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
        images, targets = batch['image'], batch['target']
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss, batch_size=len(images))
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
        images, targets = batch['image'], batch['target']
        batch_size = images.shape[0]

        outputs = self.model(images)
        loss_dict_list = self.model(images, targets)  # list of dictionaries
        total_loss = sum(
            sum(loss_item for loss_item in loss_dict.values() if loss_item.ndim == 0)
            for loss_dict in loss_dict_list
        )

        for target in targets:
            target['masks'] = (target['masks'] > 0).to(torch.uint8)
            target['boxes'] = target['boxes'].to(torch.float32)
            target['labels'] = target['labels'].to(torch.int64)

        for output in outputs:
            if 'masks' in output:
                output['masks'] = (output['masks'] > 0.5).squeeze(1).to(torch.uint8)

        self.log('val_loss', total_loss, batch_size=batch_size)

        metrics = self.val_metrics(outputs, targets)
        # Log only scalar values from metrics
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                # Cast to float if integer and compute mean
                value = value.to(torch.float32).mean()
            scalar_metrics[key] = value

        self.log_dict(scalar_metrics, batch_size=batch_size)

        # check
        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            datamodule = self.trainer.datamodule

            batch['prediction_masks'] = [output['masks'].cpu() for output in outputs]
            batch['image'] = batch['image'].cpu()

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
        images, targets = batch['image'], batch['target']
        batch_size = images.shape[0]

        outputs = self.model(images)
        loss_dict_list = self.model(
            images, targets
        )  # Compute all losses, list of dictonaries (one for every batch element)
        total_loss = sum(
            sum(loss_item for loss_item in loss_dict.values() if loss_item.ndim == 0)
            for loss_dict in loss_dict_list
        )

        for target in targets:
            target['masks'] = target['masks'].to(torch.uint8)
            target['boxes'] = target['boxes'].to(torch.float32)
            target['labels'] = target['labels'].to(torch.int64)

        for output in outputs:
            if 'masks' in output:
                output['masks'] = (output['masks'] > 0.5).squeeze(1).to(torch.uint8)

        self.log('test_loss', total_loss, batch_size=batch_size)

        metrics = self.val_metrics(outputs, targets)
        # Log only scalar values from metrics
        scalar_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                # Cast to float if integer and compute mean
                value = value.to(torch.float32).mean()
            scalar_metrics[key] = value

        self.log_dict(scalar_metrics, batch_size=batch_size)

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
        images = batch['image']

        with torch.no_grad():
            outputs = self.model(images)

        for output in outputs:
            keep = output['scores'] > 0.05
            output['boxes'] = output['boxes'][keep]
            output['labels'] = output['labels'][keep]
            output['scores'] = output['scores'][keep]
            output['masks'] = (output['masks'] > 0.5).squeeze(1).to(torch.uint8)[keep]

        return outputs
