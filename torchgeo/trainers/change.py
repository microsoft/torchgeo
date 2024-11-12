# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for change detection."""

import os
import warnings
from typing import Any, List, Optional, Union

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score,
    MulticlassJaccardIndex,
)
from torchmetrics.wrappers import ClasswiseWrapper
from torchvision.models._api import WeightsEnum

from ..models import FCSiamConc, FCSiamDiff, get_weight
from . import utils
from .base import BaseTask


class FocalJaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.focal_loss = smp.losses.FocalLoss(
            mode="multiclass", normalized=True)
        self.jaccard_loss = smp.losses.JaccardLoss(mode="multiclass")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.focal_loss(preds, targets) + self.jaccard_loss(preds, targets)


class XEntJaccardLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.jaccard_loss = smp.losses.JaccardLoss(mode="multiclass")

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_loss(preds, targets) + self.jaccard_loss(preds, targets)


class ChangeDetectionTask(BaseTask):
    """Change Detection."""

    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        num_classes: int = 2,
        class_weights: Optional[Tensor] = None,
        labels: Optional[List[str]] = None,
        loss: str = "ce-jaccard",
        ignore_index: Optional[int] = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """Inititalize a new ChangeDetectionTask instance.

        Args:
            model: Name of the model to use.
            backbone: Name of the `timm
                <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or `smp
                <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False or
                None for random weights, or the path to a saved model state dict. FCN
                model does not support pretrained weights. Pretrained ViT weight enums
                are not supported yet.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes.
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss.
            labels: Optional labels to use for classes in metrics
                e.g. ["background", "change"]
            loss: Name of the loss function, currently supports
                'ce', 'jaccard', 'focal' or 'focal-jaccard' loss.
            ignore_index: Optional integer class index to ignore in the loss and
                metrics.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head.
            freeze_decoder: Freeze the decoder network to linear probe
                the segmentation head.

        .. versionadded: 0.6
        """
        if ignore_index is not None and loss == "jaccard":
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )

        self.weights = weights
        super().__init__(ignore="weights")

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        ignore_index = self.hparams["ignore_index"]
        if loss == "ce":
            ignore_value = -1000 if ignore_index is None else ignore_index
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=self.hparams["class_weights"]
            )
        elif loss == "jaccard":
            self.criterion = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        elif loss == "focal-jaccard":
            self.criterion = FocalJaccardLoss()
        elif loss == "ce-jaccard":
            self.criterion = XEntJaccardLoss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        labels: Optional[List[str]] = self.hparams["labels"]
        metrics = MetricCollection(
            {
                "accuracy": ClasswiseWrapper(
                    MulticlassAccuracy(
                        num_classes=num_classes, ignore_index=ignore_index, average=None
                    ),
                    labels,
                ),
                "jaccard": ClasswiseWrapper(
                    MulticlassJaccardIndex(
                        num_classes=num_classes, ignore_index=ignore_index, average=None
                    ),
                    labels,
                ),
                "f1": ClasswiseWrapper(
                    MulticlassF1Score(
                        num_classes=num_classes, ignore_index=ignore_index, average=None
                    ),
                    labels,
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_models(self) -> None:
        """Initialize the model.

        Raises:
            ValueError: If *model* is invalid.
        """
        model: str = self.hparams["model"]
        backbone: str = self.hparams["backbone"]
        weights = self.weights
        in_channels: int = self.hparams["in_channels"]
        num_classes: int = self.hparams["num_classes"]

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels * 2,  # images are concatenated
                classes=num_classes,
            )
        elif model == "fcsiamdiff":
            self.model = FCSiamDiff(
                in_channels=in_channels,
                classes=num_classes,
                encoder_weights="imagenet" if weights is True else None,
            )
        elif model == "fcsiamconc":
            self.model = FCSiamConc(
                in_channels=in_channels,
                classes=num_classes,
                encoder_weights="imagenet" if weights is True else None,
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'fcsiamdiff, and 'fcsiamconc'."
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
        if self.hparams["freeze_backbone"] and model in ["unet"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model in ["unet"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def _shared_step(self, batch: Any, batch_idx: int, stage: str) -> Tensor:
        model: str = self.hparams["model"]
        x = batch["image"]
        y = batch["mask"]
        if model == "unet":
            x = x.flatten(start_dim=1, end_dim=2)
        y_hat = self(x)

        loss: Tensor = self.criterion(y_hat, y)
        self.log(f"{stage}_loss", loss)

        # Retrieve the correct metrics based on the stage
        metrics = getattr(self, f"{stage}_metrics", None)
        if metrics:
            metrics(y_hat, y)
            self.log_dict({f"{k}": v for k, v in metrics.compute().items()})

        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        loss = self._shared_step(batch, batch_idx, "train")
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch: Any, batch_idx: int) -> None:
        self._shared_step(batch, batch_idx, "test")

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted class.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted class.
        """
        model: str = self.hparams["model"]
        x = batch["image"]
        if model == "unet":
            x = x.flatten(start_dim=1, end_dim=2)
        y_hat: Tensor = self(x)
        y_hat_hard = y_hat.argmax(dim=1)
        return y_hat_hard
