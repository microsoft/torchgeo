# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""RF-DETR models."""

from typing import Any

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert

from ..datasets.utils import lazy_import


class RFDETR(nn.Module):
    """Adapt RF-DETR to the torchvision object detection model interface."""

    def __init__(
        self,
        variant: str,
        num_classes: int,
        in_channels: int,
        freeze_backbone: bool,
        **kwargs: Any,
    ) -> None:
        """Initialize an RF-DETR model.

        Args:
            variant: RF-DETR model variant.
            num_classes: Number of classes, including the background class.
            in_channels: Number of input channels.
            freeze_backbone: Freeze the model encoder.
            **kwargs: Additional RF-DETR model configuration parameters.
        """
        super().__init__()
        rfdetr_config = lazy_import('rfdetr.config')
        rfdetr_models = lazy_import('rfdetr.models')

        variants = {
            'rf-detr-nano': rfdetr_config.RFDETRNanoConfig,
            'rf-detr-small': rfdetr_config.RFDETRSmallConfig,
            'rf-detr-medium': rfdetr_config.RFDETRMediumConfig,
            'rf-detr-large': rfdetr_config.RFDETRLargeConfig,
        }
        kwargs.setdefault('num_channels', in_channels)
        kwargs.setdefault('freeze_encoder', freeze_backbone)
        model_config = variants[variant](num_classes=num_classes - 1, **kwargs)
        train_config = rfdetr_config.TrainConfig(dataset_dir='.', output_dir='.')
        self.model = rfdetr_models.build_model_from_config(model_config, train_config)
        if model_config.pretrain_weights is not None:
            rfdetr_models.load_pretrain_weights(self.model, model_config)
        self.criterion, self.postprocess = rfdetr_models.build_criterion_from_config(
            model_config, train_config
        )

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
            losses = self.criterion(outputs, targets)
            return {
                key: losses[key] * self.criterion.weight_dict[key]
                for key in losses
                if key in self.criterion.weight_dict
            }

        sizes = torch.tensor(
            [[height, width]] * len(images), device=images.device, dtype=torch.int64
        )
        predictions: list[dict[str, Tensor]] = self.postprocess(outputs, sizes)
        for prediction in predictions:
            prediction['labels'] += 1
        return predictions
