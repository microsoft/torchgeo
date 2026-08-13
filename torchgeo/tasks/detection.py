# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for object detection."""

from functools import partial
from typing import Any

import kornia.augmentation as K
import matplotlib.pyplot as plt
import torch
import torchvision.models.detection
from matplotlib.figure import Figure
from timm.models import adapt_input_conv
from torch import Tensor
from torch.nn.parameter import Parameter
from torchmetrics import MetricCollection
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models._api import WeightsEnum
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.retinanet import RetinaNetHead
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.ops import (
    MultiScaleRoIAlign,
    box_convert,
    feature_pyramid_network,
    misc,
)

from ..datamodules import BaseDataModule
from ..datasets import RGBBandsMissingError
from ..datasets.utils import Sample, lazy_import
from .base import BaseTask
from .utils import GeneralizedRCNNTransformNoOp

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


class ObjectDetection(BaseTask):
    """Object detection.

    .. versionadded:: 0.4
    """

    monitor = 'val_map'
    mode = 'max'

    def __init__(
        self,
        model: str = 'faster-rcnn',
        backbone: str = 'resnet50',
        weights: WeightsEnum | str | None = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        trainable_layers: int = 3,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize a new ObjectDetection instance.

        Note that we disable the internal normalize+resize transform of the detection models.
        Please ensure your images are appropriately resized before passing them to the model.

        Args:
            model: Name of the `torchvision
                <https://docs.pytorch.org/vision/stable/models.html#object-detection>`__
                model to use. One of 'faster-rcnn', 'fcos', 'retinanet',
                'rf-detr-nano', 'rf-detr-small', 'rf-detr-medium', or 'rf-detr-large'.
            backbone: Name of the `torchvision
                <https://docs.pytorch.org/vision/stable/models.html#classification>`__
                backbone to use. One of 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                'wide_resnet50_2', or 'wide_resnet101_2'.
            weights: Initial model weights. RF-DETR models accept a checkpoint path.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes (including the background).
            trainable_layers: Number of trainable layers.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the detection
                head.
            **kwargs: Additional model-specific keyword arguments.

        .. versionchanged:: 0.4
           *detection_model* was renamed to *model*.

        .. versionadded:: 0.5
           The *freeze_backbone* parameter.

        .. versionchanged:: 0.5
           *pretrained*, *learning_rate*, and *learning_rate_schedule_patience* were
           renamed to *weights*, *lr*, and *patience*.
        """
        self.weights = weights
        self.model_kwargs = kwargs
        super().__init__()

    def forward(
        self, images: Tensor, targets: list[dict[str, Tensor]] | None = None
    ) -> dict[str, Tensor] | list[dict[str, Tensor]]:
        """Run a forward pass.

        Args:
            images: Batched images with shape ``(B, C, H, W)``.
            targets: Ground-truth annotations for each image. Each dictionary
                contains ``boxes`` with shape ``(N, 4)`` in absolute ``xyxy``
                coordinates and one-indexed ``labels`` with shape ``(N,)``.
                If ``None``, the model runs in inference mode.

        Returns:
            A dictionary mapping loss names to scalar tensors when *targets* are
            provided. Otherwise, a list containing one prediction dictionary per
            image with absolute ``xyxy`` boxes, one-indexed labels, and scores.
        """
        if not self.hparams['model'].startswith('rf-detr'):
            return self.model(images, targets)

        height, width = images.shape[-2:]
        if targets is not None:
            scale = images.new_tensor([width, height, width, height])
            targets = [
                {
                    'boxes': box_convert(target['boxes'], 'xyxy', 'cxcywh') / scale,
                    'labels': target['labels'] - 1,
                }
                for target in targets
            ]

        outputs = self.model(images, targets)
        if targets is not None:
            losses = self.rf_detr_criterion(outputs, targets)
            return {
                key: losses[key] * self.rf_detr_criterion.weight_dict[key]
                for key in losses
                if key in self.rf_detr_criterion.weight_dict
            }

        sizes = torch.tensor(
            [[height, width]] * len(images), device=images.device, dtype=torch.int64
        )
        predictions: list[dict[str, Tensor]] = self.rf_detr_postprocess(outputs, sizes)
        for prediction in predictions:
            prediction['labels'] += 1
        return predictions

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* or *backbone* are invalid.
        """
        backbone: str = self.hparams['backbone']
        model: str = self.hparams['model']
        in_channels: int = self.hparams['in_channels']
        num_classes: int = self.hparams['num_classes']
        freeze_backbone: bool = self.hparams['freeze_backbone']

        if model.startswith('rf-detr'):
            rfdetr_config = lazy_import('rfdetr.config')
            rfdetr_models = lazy_import('rfdetr.models')

            variants = {
                'rf-detr-nano': rfdetr_config.RFDETRNanoConfig,
                'rf-detr-small': rfdetr_config.RFDETRSmallConfig,
                'rf-detr-medium': rfdetr_config.RFDETRMediumConfig,
                'rf-detr-large': rfdetr_config.RFDETRLargeConfig,
            }
            self.model_kwargs.setdefault('num_channels', in_channels)
            self.model_kwargs.setdefault('freeze_encoder', freeze_backbone)
            self.model_kwargs['pretrain_weights'] = self.weights
            model_config = variants[model](
                num_classes=num_classes - 1, **self.model_kwargs
            )
            train_config = rfdetr_config.TrainConfig(dataset_dir='.', output_dir='.')
            self.model = rfdetr_models.build_model_from_config(
                model_config, train_config
            )
            if self.weights is not None:
                if isinstance(self.weights, WeightsEnum):
                    state_dict = self.weights.get_state_dict(
                        progress=True, check_hash=True, weights_only=True
                    )
                else:
                    state_dict = torch.load(
                        self.weights, map_location='cpu', weights_only=True
                    )
                self.model.load_state_dict(state_dict)
            self.rf_detr_criterion, self.rf_detr_postprocess = (
                rfdetr_models.build_criterion_from_config(model_config, train_config)
            )
            return

        if backbone in BACKBONE_LAT_DIM_MAP:
            kwargs = {
                'backbone_name': backbone,
                'trainable_layers': self.hparams['trainable_layers'],
                'weights': self.weights,
            }
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
            self.model.transform = GeneralizedRCNNTransformNoOp()
        elif model == 'fcos':
            kwargs['extra_blocks'] = feature_pyramid_network.LastLevelP6P7(256, 256)
            kwargs['norm_layer'] = (
                misc.FrozenBatchNorm2d if self.weights else torch.nn.BatchNorm2d
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
            self.model.transform = GeneralizedRCNNTransformNoOp()
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
            self.model.transform = GeneralizedRCNNTransformNoOp()
        else:
            raise ValueError(f"Model type '{model}' is not valid.")

        weight = adapt_input_conv(in_channels, self.model.backbone.body.conv1.weight)  # ty: ignore[invalid-argument-type]
        self.model.backbone.body.conv1.weight = Parameter(weight)  # ty: ignore[invalid-assignment]
        self.model.backbone.body.conv1.in_channels = in_channels  # ty: ignore[invalid-assignment]

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
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
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
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
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
            batch['prediction_bbox_xyxy'] = [b['boxes'].cpu() for b in y_hat]
            batch['prediction_label'] = [b['labels'].cpu() for b in y_hat]
            batch['prediction_score'] = [b['scores'].cpu() for b in y_hat]
            batch['image'] = batch['image'].cpu()
            sample = {key: value[0] for key, value in batch.items()}

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
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
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
