# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for object detection."""

from functools import partial
from typing import Any

import matplotlib.pyplot as plt
import torch
import torchvision.models.detection
from matplotlib.figure import Figure
from timm.models import adapt_input_conv
from torch import Tensor
from torch.nn.parameter import Parameter
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import resnet as R
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign, feature_pyramid_network, misc

from ..datasets import RGBBandsMissingError, unbind_samples
from .base import BaseTask

BACKBONE_LAT_DIM_MAP = {
    'resnet18': 512,
    'resnet34': 512,
    'resnet50': 2048,
    'resnet101': 2048,
    'resnet152': 2048,
    'resnext50_32x4d': 2048,
    'resnext101_32x8d': 2048,
    'wide_resnet50_2': 2048,
    'wide_resnet101_2': 2048,
}

BACKBONE_WEIGHT_MAP = {
    'resnet18': R.ResNet18_Weights.DEFAULT,
    'resnet34': R.ResNet34_Weights.DEFAULT,
    'resnet50': R.ResNet50_Weights.DEFAULT,
    'resnet101': R.ResNet101_Weights.DEFAULT,
    'resnet152': R.ResNet152_Weights.DEFAULT,
    'resnext50_32x4d': R.ResNeXt50_32X4D_Weights.DEFAULT,
    'resnext101_32x8d': R.ResNeXt101_32X8D_Weights.DEFAULT,
    'wide_resnet50_2': R.Wide_ResNet50_2_Weights.DEFAULT,
    'wide_resnet101_2': R.Wide_ResNet101_2_Weights.DEFAULT,
}


class ObjectDetectionTask(BaseTask):
    """Object detection.

    .. versionadded:: 0.4
    """

    ignore = None
    monitor = 'val_map'
    mode = 'max'

    def __init__(
        self,
        model: str = 'faster-rcnn',
        backbone: str = 'resnet50',
        weights: bool | None = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        trainable_layers: int = 3,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize a new ObjectDetectionTask instance.

        Args:
            model: Name of the `torchvision
                <https://pytorch.org/vision/stable/models.html#object-detection>`__
                model to use. One of 'faster-rcnn', 'fcos', or 'retinanet'.
            backbone: Name of the `torchvision
                <https://pytorch.org/vision/stable/models.html#classification>`__
                backbone to use. One of 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                'wide_resnet50_2', or 'wide_resnet101_2'.
            weights: Initial model weights. True for ImageNet weights, False or None
                for random weights.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes (including the background).
            trainable_layers: Number of trainable layers.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the detection
                head.

        .. versionchanged:: 0.4
           *detection_model* was renamed to *model*.

        .. versionadded:: 0.5
           The *freeze_backbone* parameter.

        .. versionchanged:: 0.5
           *pretrained*, *learning_rate*, and *learning_rate_schedule_patience* were
           renamed to *weights*, *lr*, and *patience*.
        """
        super().__init__()

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* or *backbone* are invalid.
        """
        backbone: str = self.hparams['backbone']
        model: str = self.hparams['model']
        weights: bool | None = self.hparams['weights']
        in_channels: int = self.hparams['in_channels']
        num_classes: int = self.hparams['num_classes']
        freeze_backbone: bool = self.hparams['freeze_backbone']

        if backbone in BACKBONE_LAT_DIM_MAP:
            kwargs = {
                'backbone_name': backbone,
                'trainable_layers': self.hparams['trainable_layers'],
            }
            if weights:
                kwargs['weights'] = BACKBONE_WEIGHT_MAP[backbone]
            else:
                kwargs['weights'] = None

            latent_dim = BACKBONE_LAT_DIM_MAP[backbone]
        else:
            raise ValueError(f"Backbone type '{backbone}' is not valid.")

        if model == 'faster-rcnn':
            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((32), (64), (128), (256), (512)), aspect_ratios=((0.5, 1.0, 2.0))
            )

            roi_pooler = MultiScaleRoIAlign(
                featmap_names=['0', '1', '2', '3'], output_size=7, sampling_ratio=2
            )

            if freeze_backbone:
                for param in model_backbone.parameters():
                    param.requires_grad = False

            self.model = torchvision.models.detection.FasterRCNN(
                model_backbone,
                num_classes,
                rpn_anchor_generator=anchor_generator,
                box_roi_pool=roi_pooler,
            )
        elif model == 'fcos':
            kwargs['extra_blocks'] = feature_pyramid_network.LastLevelP6P7(256, 256)
            kwargs['norm_layer'] = (
                misc.FrozenBatchNorm2d if weights else torch.nn.BatchNorm2d
            )

            model_backbone = resnet_fpn_backbone(**kwargs)
            anchor_generator = AnchorGenerator(
                sizes=((8,), (16,), (32,), (64,), (128,), (256,)),
                aspect_ratios=((1.0,), (1.0,), (1.0,), (1.0,), (1.0,), (1.0,)),
            )

            if freeze_backbone:
                for param in model_backbone.parameters():
                    param.requires_grad = False

            self.model = torchvision.models.detection.FCOS(
                model_backbone, num_classes, anchor_generator=anchor_generator
            )
        elif model == 'retinanet':
            kwargs['extra_blocks'] = feature_pyramid_network.LastLevelP6P7(
                latent_dim, 256
            )
            model_backbone = resnet_fpn_backbone(**kwargs)

            anchor_sizes = (
                (16, 20, 25),
                (32, 40, 50),
                (64, 80, 101),
                (128, 161, 203),
                (256, 322, 406),
                (512, 645, 812),
            )
            aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
            anchor_generator = AnchorGenerator(anchor_sizes, aspect_ratios)

            head = RetinaNetHead(
                model_backbone.out_channels,
                anchor_generator.num_anchors_per_location()[0],
                num_classes,
                norm_layer=partial(torch.nn.GroupNorm, 32),
            )

            if freeze_backbone:
                for param in model_backbone.parameters():
                    param.requires_grad = False

            self.model = torchvision.models.detection.RetinaNet(
                model_backbone,
                num_classes,
                anchor_generator=anchor_generator,
                head=head,
            )
        else:
            raise ValueError(f"Model type '{model}' is not valid.")

        weight = adapt_input_conv(in_channels, self.model.backbone.body.conv1.weight)
        self.model.backbone.body.conv1.weight = Parameter(weight)
        self.model.backbone.body.conv1.in_channels = in_channels

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
        metrics = MetricCollection([MeanAveragePrecision(average='macro')])
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
        batch_size = x.shape[0]
        assert 'bbox_xyxy' in batch, 'bbox_xyxy is required for object detection.'
        y = [
            {'boxes': batch['bbox_xyxy'][i], 'labels': batch['label'][i]}
            for i in range(batch_size)
        ]
        loss_dict = self(x, y)
        train_loss: Tensor = sum(loss_dict.values())
        self.log_dict(loss_dict, batch_size=batch_size)
        return train_loss

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
        batch_size = x.shape[0]
        assert 'bbox_xyxy' in batch, 'bbox_xyxy is required for object detection.'
        y = [
            {'boxes': batch['bbox_xyxy'][i], 'labels': batch['label'][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x)
        metrics = self.val_metrics(y_hat, y)

        # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
        metrics.pop('val_classes', None)

        self.log_dict(metrics, batch_size=batch_size)

        if (
            batch_idx < 10
            and hasattr(self.trainer, 'datamodule')
            and hasattr(self.trainer.datamodule, 'plot')
            and self.logger
            and hasattr(self.logger, 'experiment')
            and hasattr(self.logger.experiment, 'add_figure')
        ):
            datamodule = self.trainer.datamodule
            batch['prediction_bbox_xyxy'] = [b['boxes'].cpu() for b in y_hat]
            batch['prediction_label'] = [b['labels'].cpu() for b in y_hat]
            batch['prediction_score'] = [b['scores'].cpu() for b in y_hat]
            batch['image'] = batch['image'].cpu()
            sample = unbind_samples(batch)[0]
            # Convert image to uint8 for plotting
            if torch.is_floating_point(sample['image']):
                sample['image'] *= 255
                sample['image'] = sample['image'].to(torch.uint8)

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
        batch_size = x.shape[0]
        assert 'bbox_xyxy' in batch, 'bbox_xyxy is required for object detection.'
        y = [
            {'boxes': batch['bbox_xyxy'][i], 'labels': batch['label'][i]}
            for i in range(batch_size)
        ]
        y_hat = self(x)
        metrics = self.test_metrics(y_hat, y)

        # https://github.com/Lightning-AI/torchmetrics/pull/1832#issuecomment-1623890714
        metrics.pop('test_classes', None)

        self.log_dict(metrics, batch_size=batch_size)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> list[dict[str, Tensor]]:
        """Compute the predicted bounding boxes.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch['image']
        y_hat: list[dict[str, Tensor]] = self(x)
        return y_hat
