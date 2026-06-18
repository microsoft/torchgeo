# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Trainers for object detection."""

from functools import partial

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
from torchvision.ops import MultiScaleRoIAlign, feature_pyramid_network, misc

from ..datamodules import BaseDataModule
from ..datasets import RGBBandsMissingError, unbind_samples
from ..datasets.utils import Sample, boxes_to_points
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


class ObjectDetectionTask(BaseTask):
    """Object detection.

    .. versionadded:: 0.4
    """

    monitor = 'val_map'
    mode = 'max'

    def __init__(
        self,
        model: str = 'faster-rcnn',
        backbone: str = 'resnet50',
        weights: WeightsEnum | None = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        trainable_layers: int = 3,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize a new ObjectDetectionTask instance.

        Note that we disable the internal normalize+resize transform of the detection models.
        Please ensure your images are appropriately resized before passing them to the model.

        Args:
            model: Name of the `torchvision
                <https://docs.pytorch.org/vision/stable/models.html#object-detection>`__
                model to use. One of 'faster-rcnn', 'fcos', or 'retinanet'.
            backbone: Name of the `torchvision
                <https://docs.pytorch.org/vision/stable/models.html#classification>`__
                backbone to use. One of 'resnet18', 'resnet34', 'resnet50',
                'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
                'wide_resnet50_2', or 'wide_resnet101_2'.
            weights: Initial model weights.
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
        self.weights = weights
        super().__init__()

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


class PointDetectionTask(ObjectDetectionTask):
    """Point detection using object detection proxy boxes.

    Point annotations are trained as fixed-size proxy boxes. At prediction and
    evaluation time, predicted boxes are collapsed back to center points.

    .. versionadded:: 0.10
    """

    monitor = 'val_point_f1'
    mode = 'max'

    def __init__(
        self,
        model: str = 'faster-rcnn',
        backbone: str = 'resnet50',
        weights: WeightsEnum | None = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        trainable_layers: int = 3,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        distance_threshold: float = 20.0,
        score_threshold: float = 0.5,
    ) -> None:
        """Initialize a new PointDetectionTask instance.

        Args:
            model: Name of the torchvision object detection model to use.
            backbone: Name of the torchvision backbone to use.
            weights: Initial model weights.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes (including the background).
            trainable_layers: Number of trainable layers.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the detection
                head.
            distance_threshold: Maximum pixel distance for matching a predicted point
                to a target point.
            score_threshold: Minimum prediction score used for point metrics.

        Raises:
            ValueError: If *distance_threshold* or *score_threshold* is invalid.
        """
        if distance_threshold <= 0:
            raise ValueError('distance_threshold must be positive')
        if not 0 <= score_threshold <= 1:
            raise ValueError('score_threshold must be in the range [0, 1]')

        super().__init__(
            model=model,
            backbone=backbone,
            weights=weights,
            in_channels=in_channels,
            num_classes=num_classes,
            trainable_layers=trainable_layers,
            lr=lr,
            patience=patience,
            freeze_backbone=freeze_backbone,
        )

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """Compute the validation metrics."""
        x = batch['image']
        batch_size = x.shape[0]
        y_hat = self(x)
        metrics = self._point_metrics(batch, y_hat, prefix='val_')
        self.log_dict(metrics, batch_size=batch_size)

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test metrics."""
        x = batch['image']
        batch_size = x.shape[0]
        y_hat = self(x)
        metrics = self._point_metrics(batch, y_hat, prefix='test_')
        self.log_dict(metrics, batch_size=batch_size)

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> list[dict[str, Tensor]]:
        """Compute predicted boxes and center points."""
        y_hat = super().predict_step(batch, batch_idx, dataloader_idx)
        return self._add_prediction_points(y_hat)

    def _add_prediction_points(
        self, predictions: list[dict[str, Tensor]]
    ) -> list[dict[str, Tensor]]:
        """Add center points to object detection predictions."""
        outputs: list[dict[str, Tensor]] = []
        for prediction in predictions:
            output = dict(prediction)
            output['points'] = boxes_to_points(prediction['boxes'])
            outputs.append(output)
        return outputs

    def _point_metrics(
        self, batch: Sample, predictions: list[dict[str, Tensor]], prefix: str
    ) -> dict[str, Tensor]:
        """Compute point detection TP, FP, FN, precision, recall, and F1."""
        device = batch['image'].device
        score_threshold: float = self.hparams['score_threshold']
        distance_threshold: float = self.hparams['distance_threshold']

        total_tp = torch.tensor(0.0, device=device)
        total_fp = torch.tensor(0.0, device=device)
        total_fn = torch.tensor(0.0, device=device)

        for i, prediction in enumerate(predictions):
            prediction = self._prediction_to_points(prediction, score_threshold)
            pred_points = prediction['points']
            pred_labels = prediction['labels']

            target_points = self._target_points(batch, i).to(device)
            target_labels = batch['label'][i].to(device)

            matched_pred, unmatched_pred, unmatched_gt = self._match_points(
                pred_points,
                pred_labels,
                target_points,
                target_labels,
                distance_threshold,
            )
            total_tp += matched_pred.numel()
            total_fp += unmatched_pred.numel()
            total_fn += unmatched_gt.numel()

        precision = total_tp / (total_tp + total_fp).clamp(min=1)
        recall = total_tp / (total_tp + total_fn).clamp(min=1)
        f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)

        return {
            f'{prefix}point_tp': total_tp,
            f'{prefix}point_fp': total_fp,
            f'{prefix}point_fn': total_fn,
            f'{prefix}point_precision': precision,
            f'{prefix}point_recall': recall,
            f'{prefix}point_f1': f1,
        }

    def _prediction_to_points(
        self, prediction: dict[str, Tensor], score_threshold: float
    ) -> dict[str, Tensor]:
        """Convert an object detector prediction to point detections."""
        scores = prediction.get(
            'scores',
            torch.ones(
                (prediction['boxes'].shape[0],),
                dtype=prediction['boxes'].dtype,
                device=prediction['boxes'].device,
            ),
        )
        keep = scores >= score_threshold
        if 'points' in prediction:
            points = prediction['points']
        else:
            points = boxes_to_points(prediction['boxes'])

        return {
            'points': points[keep],
            'scores': scores[keep],
            'labels': prediction['labels'][keep],
        }

    def _target_points(self, batch: Sample, index: int) -> Tensor:
        """Get target points from a batch."""
        if 'points' in batch:
            return batch['points'][index]
        return boxes_to_points(batch['bbox_xyxy'][index])

    def _match_points(
        self,
        pred_points: Tensor,
        pred_labels: Tensor,
        target_points: Tensor,
        target_labels: Tensor,
        distance_threshold: float,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """One-to-one match points by class and distance."""
        device = pred_points.device
        pred_count = pred_points.shape[0]
        target_count = target_points.shape[0]

        if pred_count == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
                torch.arange(target_count, dtype=torch.int64, device=device),
            )
        if target_count == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.arange(pred_count, dtype=torch.int64, device=device),
                torch.empty(0, dtype=torch.int64, device=device),
            )

        distances = torch.cdist(pred_points.float(), target_points.float())
        same_class = pred_labels[:, None] == target_labels[None, :]
        candidate_pred, candidate_target = torch.where(
            same_class & (distances <= distance_threshold)
        )
        if candidate_pred.numel() == 0:
            return (
                torch.empty(0, dtype=torch.int64, device=device),
                torch.arange(pred_count, dtype=torch.int64, device=device),
                torch.arange(target_count, dtype=torch.int64, device=device),
            )

        candidate_distances = distances[candidate_pred, candidate_target]
        order = torch.argsort(candidate_distances)
        used_pred: set[int] = set()
        used_target: set[int] = set()
        matched_pred: list[int] = []

        for idx in order.tolist():
            pred_idx = int(candidate_pred[idx])
            target_idx = int(candidate_target[idx])
            if pred_idx in used_pred or target_idx in used_target:
                continue
            used_pred.add(pred_idx)
            used_target.add(target_idx)
            matched_pred.append(pred_idx)

        unmatched_pred = [idx for idx in range(pred_count) if idx not in used_pred]
        unmatched_target = [
            idx for idx in range(target_count) if idx not in used_target
        ]

        return (
            torch.tensor(matched_pred, dtype=torch.int64, device=device),
            torch.tensor(unmatched_pred, dtype=torch.int64, device=device),
            torch.tensor(unmatched_target, dtype=torch.int64, device=device),
        )
