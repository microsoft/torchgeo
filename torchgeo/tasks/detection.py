# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tasks for object detection."""

import math
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
from torchvision.ops import MultiScaleRoIAlign, feature_pyramid_network, misc

from ..datamodules import BaseDataModule
from ..datasets import RGBBandsMissingError, unbind_samples
from ..datasets.utils import Sample
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

RF_DETR_MODEL_CONFIG_MAP = {
    'rf-detr-nano': 'RFDETRNanoConfig',
    'rf-detr-small': 'RFDETRSmallConfig',
    'rf-detr-medium': 'RFDETRMediumConfig',
    'rf-detr-large': 'RFDETRLargeConfig',
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
        weights: WeightsEnum | None = None,
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
                'rf-detr-nano', 'rf-detr-small', 'rf-detr-medium', or
                'rf-detr-large'.
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
            **kwargs: Additional model-specific keyword arguments. When ``model`` is
                an RF-DETR variant, these are split between RF-DETR's model config
                (for example ``resolution`` or ``pretrain_weights``) and train config
                (for example ``lr_encoder`` or ``weight_decay``).

        .. versionchanged:: 0.4
           *detection_model* was renamed to *model*.

        .. versionadded:: 0.5
           The *freeze_backbone* parameter.

        .. versionchanged:: 0.5
           *pretrained*, *learning_rate*, and *learning_rate_schedule_patience* were
           renamed to *weights*, *lr*, and *patience*.
        """
        self.weights = weights
        self.rf_detr_kwargs = dict(kwargs)
        self.rf_detr_model_config: Any | None = None
        self.rf_detr_train_config: Any | None = None
        self.rf_detr_criterion: Any | None = None
        self.rf_detr_postprocess: Any | None = None
        self._rf_detr_runtime_error: ImportError | None = None
        self._rf_detr_runtime_ready = False
        super().__init__()
        self.hparams['kwargs'] = self.rf_detr_kwargs

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model."""
        if self._use_rf_detr_backend():
            self._ensure_rf_detr_runtime()
        return super().forward(*args, **kwargs)

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
        rf_detr_kwargs = dict(self.rf_detr_kwargs)

        if model in RF_DETR_MODEL_CONFIG_MAP:
            self._configure_rf_detr_model(
                model=model,
                backbone=backbone,
                in_channels=in_channels,
                num_classes=num_classes,
                freeze_backbone=freeze_backbone,
                rf_detr_kwargs=rf_detr_kwargs,
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

    @staticmethod
    def _missing_rf_detr_dependency_error() -> ImportError:
        return ImportError(
            "RF-DETR support requires the optional 'rfdetr' dependency. "
            'Install it with `pip install rfdetr`.'
        )

    @staticmethod
    def _incompatible_rf_detr_runtime_error() -> ImportError:
        return ImportError(
            'RF-DETR runtime could not be imported. Ensure `rfdetr` and '
            '`transformers` are compatible; for example, `rfdetr>=1.6` '
            'requires `transformers>=5.1`.'
        )

    @staticmethod
    def _load_rf_detr_config_dependencies() -> tuple[Any, Any, Any, Any, Any, Any]:
        from rfdetr.config import (
            ModelConfig,
            RFDETRLargeConfig,
            RFDETRMediumConfig,
            RFDETRNanoConfig,
            RFDETRSmallConfig,
            TrainConfig,
        )

        return (
            ModelConfig,
            RFDETRLargeConfig,
            RFDETRMediumConfig,
            RFDETRNanoConfig,
            RFDETRSmallConfig,
            TrainConfig,
        )

    @staticmethod
    def _load_rf_detr_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
        from rfdetr._namespace import build_namespace
        from rfdetr.models.lwdetr import build_criterion_and_postprocessors, build_model
        from rfdetr.models.weights import load_pretrain_weights

        return (
            build_namespace,
            build_criterion_and_postprocessors,
            build_model,
            load_pretrain_weights,
        )

    def _initialize_rf_detr_runtime(self) -> None:
        """Load and build the RF-DETR runtime on demand."""
        if self._rf_detr_runtime_ready:
            return

        assert self.rf_detr_model_config is not None
        assert self.rf_detr_train_config is not None

        try:
            (
                build_namespace,
                build_criterion_and_postprocessors,
                build_model,
                load_pretrain_weights,
            ) = self._load_rf_detr_runtime_dependencies()
        except ModuleNotFoundError as exc:
            error = self._missing_rf_detr_dependency_error()
            self._rf_detr_runtime_error = error
            raise error from exc
        except ImportError as exc:
            error = self._incompatible_rf_detr_runtime_error()
            self._rf_detr_runtime_error = error
            raise error from exc

        namespace = build_namespace(
            self.rf_detr_model_config, self.rf_detr_train_config
        )
        self.model = build_model(namespace)
        if self.rf_detr_model_config.pretrain_weights is not None:
            load_pretrain_weights(self.model, self.rf_detr_model_config)
            namespace = build_namespace(
                self.rf_detr_model_config, self.rf_detr_train_config
            )
        self.rf_detr_criterion, self.rf_detr_postprocess = (
            build_criterion_and_postprocessors(namespace)
        )
        self._rf_detr_runtime_error = None
        self._rf_detr_runtime_ready = True

    def _ensure_rf_detr_runtime(self) -> None:
        """Ensure RF-DETR runtime dependencies are available before use."""
        if not self._use_rf_detr_backend():
            return
        if self._rf_detr_runtime_error is not None:
            raise self._rf_detr_runtime_error
        self._initialize_rf_detr_runtime()

    def _configure_rf_detr_model(
        self,
        model: str,
        backbone: str,
        in_channels: int,
        num_classes: int,
        freeze_backbone: bool,
        rf_detr_kwargs: dict[str, Any],
    ) -> None:
        """Initialize an RF-DETR model and its training utilities."""
        if backbone != 'resnet50':
            raise ValueError(
                'Backbone selection is not supported for RF-DETR. '
                "Leave backbone='resnet50' and use RF-DETR kwargs instead."
            )
        if self.weights is not None:
            raise ValueError(
                "The 'weights' argument is not supported for RF-DETR. "
                'Use RF-DETR kwargs such as pretrain_weights=... instead.'
            )
        if in_channels != 3:
            raise ValueError('RF-DETR currently requires in_channels=3.')
        if num_classes < 2:
            raise ValueError(
                "RF-DETR requires num_classes >= 2 when using TorchGeo's "
                'num_classes API, which includes background.'
            )

        try:
            (
                ModelConfig,
                RFDETRLargeConfig,
                RFDETRMediumConfig,
                RFDETRNanoConfig,
                RFDETRSmallConfig,
                TrainConfig,
            ) = self._load_rf_detr_config_dependencies()
        except ModuleNotFoundError as exc:
            raise self._missing_rf_detr_dependency_error() from exc

        config_map = {
            'rf-detr-nano': RFDETRNanoConfig,
            'rf-detr-small': RFDETRSmallConfig,
            'rf-detr-medium': RFDETRMediumConfig,
            'rf-detr-large': RFDETRLargeConfig,
        }
        model_fields = set(ModelConfig.model_fields)
        train_fields = set(TrainConfig.model_fields)
        model_kwargs: dict[str, Any] = {}
        train_kwargs: dict[str, Any] = {}

        for key, value in rf_detr_kwargs.items():
            if key in model_fields:
                model_kwargs[key] = value
            elif key in train_fields:
                train_kwargs[key] = value
            else:
                allowed = ', '.join(sorted(model_fields | train_fields))
                raise ValueError(
                    f"Unknown RF-DETR parameter '{key}'. Available RF-DETR "
                    f'parameter(s): {allowed}.'
                )

        if 'num_classes' in model_kwargs:
            raise ValueError(
                'Do not pass num_classes through RF-DETR kwargs. '
                'Use ObjectDetectionTask(num_classes=...), which TorchGeo '
                'interprets as including background.'
            )

        model_kwargs.setdefault('num_classes', num_classes - 1)
        model_kwargs.setdefault('pretrain_weights', None)
        if freeze_backbone:
            model_kwargs.setdefault('freeze_encoder', True)

        train_kwargs.setdefault('dataset_dir', '.')
        train_kwargs.setdefault('output_dir', '.')
        train_kwargs.setdefault('lr', self.hparams['lr'])

        model_config_class = config_map[model]
        self.rf_detr_model_config = model_config_class(**model_kwargs)
        self.rf_detr_train_config = TrainConfig(**train_kwargs)
        self.model = torch.nn.Identity()
        try:
            self._initialize_rf_detr_runtime()
        except ImportError:
            pass

    def _use_rf_detr_backend(self) -> bool:
        """Return whether the current task is backed by RF-DETR."""
        return self.hparams['model'] in RF_DETR_MODEL_CONFIG_MAP

    def _build_targets(self, batch: Sample, batch_size: int) -> list[dict[str, Tensor]]:
        """Build torchvision-style targets from a TorchGeo batch."""
        return [
            {'boxes': batch['bbox_xyxy'][i], 'labels': batch['label'][i]}
            for i in range(batch_size)
        ]

    def _build_rf_detr_batch(
        self, batch: Sample
    ) -> tuple[Any, list[dict[str, Tensor]], list[dict[str, Tensor]]]:
        """Convert a TorchGeo batch into RF-DETR inputs and targets."""
        self._ensure_rf_detr_runtime()
        from rfdetr.utilities.tensors import nested_tensor_from_tensor_list

        images = list(batch['image'].unbind())
        targets: list[dict[str, Tensor]] = []
        metric_targets: list[dict[str, Tensor]] = []

        has_boxes = 'bbox_xyxy' in batch
        for index, image in enumerate(images):
            height, width = image.shape[-2:]
            orig_size = torch.tensor(
                [height, width], dtype=torch.int64, device=image.device
            )
            target: dict[str, Tensor] = {
                'image_id': torch.tensor(
                    [index], dtype=torch.int64, device=image.device
                ),
                'orig_size': orig_size,
                'size': orig_size,
            }

            if has_boxes:
                boxes_xyxy = batch['bbox_xyxy'][index].float()
                labels = batch['label'][index].long()
                if len(labels) > 0 and torch.any(labels < 1):
                    raise ValueError(
                        'TorchGeo RF-DETR support expects foreground labels to start at 1 '
                        'because num_classes includes the background class.'
                    )

                boxes_cxcywh = torch.zeros_like(boxes_xyxy)
                if boxes_xyxy.numel() > 0:
                    boxes_cxcywh[:, 0] = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2
                    boxes_cxcywh[:, 1] = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2
                    boxes_cxcywh[:, 2] = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
                    boxes_cxcywh[:, 3] = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
                    scale = torch.tensor(
                        [width, height, width, height],
                        dtype=boxes_xyxy.dtype,
                        device=boxes_xyxy.device,
                    )
                    boxes_cxcywh = boxes_cxcywh / scale

                area = (boxes_xyxy[:, 2] - boxes_xyxy[:, 0]) * (
                    boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
                )
                target.update(
                    {
                        'boxes': boxes_cxcywh,
                        'labels': labels - 1,
                        'area': area,
                        'iscrowd': torch.zeros_like(labels, dtype=torch.int64),
                    }
                )
                metric_targets.append({'boxes': boxes_xyxy, 'labels': labels})

            targets.append(target)

        samples = nested_tensor_from_tensor_list(images)
        return samples, targets, metric_targets

    def _postprocess_rf_detr(
        self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]]
    ) -> list[dict[str, Tensor]]:
        """Convert RF-DETR outputs into TorchMetrics-compatible predictions."""
        assert self.rf_detr_postprocess is not None
        assert self.rf_detr_model_config is not None
        orig_sizes = torch.stack([target['orig_size'] for target in targets])
        predictions = self.rf_detr_postprocess(outputs, orig_sizes)
        converted_predictions: list[dict[str, Tensor]] = []
        for prediction in predictions:
            keep = prediction['labels'] < self.rf_detr_model_config.num_classes
            converted_predictions.append(
                {
                    'boxes': prediction['boxes'][keep],
                    'labels': prediction['labels'][keep] + 1,
                    'scores': prediction['scores'][keep],
                }
            )
        return converted_predictions

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
        if self._use_rf_detr_backend():
            samples, targets, _ = self._build_rf_detr_batch(batch)
            outputs = self.model(samples, targets)
            assert self.rf_detr_criterion is not None
            loss_dict = self.rf_detr_criterion(outputs, targets)
            weight_dict = self.rf_detr_criterion.weight_dict
            train_loss = sum(
                loss_dict[key] * weight_dict[key]
                for key in loss_dict
                if key in weight_dict
            )
        else:
            y = self._build_targets(batch, batch_size)
            loss_dict = self(x, y)
            train_loss = sum(loss_dict.values())
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
        if self._use_rf_detr_backend():
            samples, targets, y = self._build_rf_detr_batch(batch)
            outputs = self.model(samples)
            y_hat = self._postprocess_rf_detr(outputs, targets)
        else:
            y = self._build_targets(batch, batch_size)
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
        if self._use_rf_detr_backend():
            samples, targets, y = self._build_rf_detr_batch(batch)
            outputs = self.model(samples)
            y_hat = self._postprocess_rf_detr(outputs, targets)
        else:
            y = self._build_targets(batch, batch_size)
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
        if self._use_rf_detr_backend():
            samples, targets, _ = self._build_rf_detr_batch(batch)
            outputs = self.model(samples)
            return self._postprocess_rf_detr(outputs, targets)

        y_hat: list[dict[str, Tensor]] = self(x)
        return y_hat

    def configure_optimizers(self) -> Any:
        """Initialize the optimizer and learning rate scheduler."""
        if not self._use_rf_detr_backend():
            return super().configure_optimizers()

        self._ensure_rf_detr_runtime()
        assert self.rf_detr_model_config is not None
        assert self.rf_detr_train_config is not None

        from rfdetr._namespace import build_namespace
        from rfdetr.training.param_groups import get_param_dict

        namespace = build_namespace(
            self.rf_detr_model_config, self.rf_detr_train_config
        )
        model_for_params = getattr(self.model, '_orig_mod', self.model)
        param_dicts = [
            param_dict
            for param_dict in get_param_dict(namespace, model_for_params)
            if param_dict['params'].requires_grad
        ]
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=self.rf_detr_train_config.lr,
            weight_decay=self.rf_detr_train_config.weight_decay,
        )

        total_steps = int(self.trainer.estimated_stepping_batches)
        steps_per_epoch = max(1, total_steps // self.rf_detr_train_config.epochs)
        warmup_steps = int(steps_per_epoch * self.rf_detr_train_config.warmup_epochs)

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            if self.rf_detr_train_config.lr_scheduler == 'cosine':
                progress = float(current_step - warmup_steps) / float(
                    max(1, total_steps - warmup_steps)
                )
                return self.rf_detr_train_config.lr_min_factor + (
                    1 - self.rf_detr_train_config.lr_min_factor
                ) * 0.5 * (1 + math.cos(math.pi * progress))
            if current_step < self.rf_detr_train_config.lr_drop * steps_per_epoch:
                return 1.0
            return 0.1

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'},
        }
