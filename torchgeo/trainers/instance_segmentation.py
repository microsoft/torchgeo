# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for instance segmentation."""

from typing import Any

import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
from matplotlib.figure import Figure
from timm.models import adapt_input_conv
from torch import Tensor
from torch.nn.parameter import Parameter
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)

from ..datasets import RGBBandsMissingError, unbind_samples
from .base import BaseTask


class InstanceSegmentationTask(BaseTask):
    """Instance Segmentation.

    .. versionadded:: 0.7
    """

    ignore = None
    monitor = 'val_segm_map'
    mode = 'max'

    def __init__(
        self,
        model: str = 'mask-rcnn',
        backbone: str = 'resnet50',
        weights: bool | None = None,
        in_channels: int = 3,
        num_classes: int = 91,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize a new InstanceSegmentationTask instance.

        Args:
            model: Name of the model to use.
            backbone: Name of the backbone to use.
            weights: Initial model weights. True for ImageNet weights, False or None
                for random weights.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes (including the background).
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.
        """
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* or *backbone* are invalid.
        """
        model: str = self.hparams['model']
        backbone: str = self.hparams['backbone']
        in_channels: int = self.hparams['in_channels']
        num_classes: int = self.hparams['num_classes']

        weights = None
        weights_backbone = None
        if self.hparams['weights']:
            weights_backbone = ResNet50_Weights.IMAGENET1K_V1
            # TODO: drop last layer of weights
            if num_classes == 91:
                weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1

        # Create model
        if model == 'mask-rcnn':
            if backbone == 'resnet50':
                self.model = maskrcnn_resnet50_fpn(
                    weights=weights,
                    num_classes=num_classes,
                    weights_backbone=weights_backbone,
                )
            else:
                msg = f"Invalid backbone type '{backbone}'. Supported backbone: 'resnet50'"
                raise ValueError(msg)
        else:
            msg = f"Invalid model type '{model}'. Supported model: 'mask-rcnn'"
            raise ValueError(msg)

        weight = adapt_input_conv(in_channels, self.model.backbone.body.conv1.weight)
        self.model.backbone.body.conv1.weight = Parameter(weight)
        self.model.backbone.body.conv1.in_channels = in_channels

        # Freeze backbone
        if self.hparams['freeze_backbone']:
            for param in self.model.backbone.parameters():
                param.requires_grad = False

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * :class:`~torchmetrics.detection.mean_ap.MeanAveragePrecision`: Mean average
          precision (mAP) and mean average recall (mAR). Precision is the number of
          true positives divided by the number of true positives + false positives.
          Recall is the number of true positives divived by the number of true positives
          + false negatives. Uses 'macro' averaging. Higher values are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not
             reflect minority class accuracy.
           * 'Macro' averaging gives equal weight to each class, and is useful for
             balanced performance assessment across imbalanced classes.
        """
        metrics = MetricCollection([MeanAveragePrecision(iou_type=('bbox', 'segm'))])
        self.val_metrics = metrics.clone(prefix='val_')
        self.test_metrics = metrics.clone(prefix='test_')

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        x = batch['image']
        y = {
            'boxes': batch['bbox_xyxy'],
            'labels': batch['label'],
            'masks': batch['mask'],
        }
        loss_dict = self(x.unbind(), unbind_samples(y))
        self.log_dict(loss_dict, batch_size=len(x))
        loss: Tensor = sum(loss_dict.values())
        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = {
            'boxes': batch['bbox_xyxy'],
            'labels': batch['label'],
            'masks': batch['mask'],
        }
        y_hat = self(x.unbind())
        for pred in y_hat:
            pred['masks'] = (pred['masks'] > 0.5).squeeze(1).to(torch.uint8)

        metrics = self.val_metrics(y_hat, unbind_samples(y))

        # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
        metrics.pop('val_classes', None)

        self.log_dict(metrics, batch_size=len(x))

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

            batch['prediction_bbox_xyxy'] = [pred['boxes'].cpu() for pred in y_hat]
            batch['prediction_mask'] = [pred['masks'].cpu() for pred in y_hat]
            batch['prediction_label'] = [pred['labels'].cpu() for pred in y_hat]
            batch['prediction_score'] = [pred['scores'].cpu() for pred in y_hat]
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
        """Compute the test metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch['image']
        y = {
            'boxes': batch['bbox_xyxy'],
            'labels': batch['label'],
            'masks': batch['mask'],
        }
        y_hat = self(x.unbind())
        for pred in y_hat:
            pred['masks'] = (pred['masks'] > 0.5).squeeze(1).to(torch.uint8)

        metrics = self.test_metrics(y_hat, unbind_samples(y))

        # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
        metrics.pop('test_classes', None)

        self.log_dict(metrics, batch_size=len(x))

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> list[dict[str, Tensor]]:
        """Compute the predicted masks.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted masks.
        """
        x = batch['image']
        y_hat: list[dict[str, Tensor]] = self(x.unbind())

        for pred in y_hat:
            keep = pred['scores'] > 0.05
            pred['boxes'] = pred['boxes'][keep]
            pred['labels'] = pred['labels'][keep]
            pred['scores'] = pred['scores'][keep]
            pred['masks'] = (pred['masks'] > 0.5).squeeze(1).to(torch.uint8)[keep]

        return y_hat
