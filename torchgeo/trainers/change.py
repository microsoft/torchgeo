# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for change detection."""

import os
import warnings
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryJaccardIndex,
    MulticlassAccuracy,
    MulticlassJaccardIndex,
)
from torchvision.models._api import WeightsEnum

from ..datasets.utils import unbind_samples
from ..models import ChangeMixin, ChangeStar, ChangeStarFarSeg, FCSiamDiff, get_weight
from . import utils
from .base import BaseTask


class ChangeDetectionTask(BaseTask):
    """Change Detection."""

    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        num_classes: int = 1,
        loss: str = "ce",
        class_weights: Optional[Tensor] = None,
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
            loss: Name of the loss function, currently supports
                'ce', 'jaccard' or 'focal' loss.
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

        Warns:
            UserWarning: When loss='jaccard' and ignore_index is specified.

        .. versionadded: 0.6
        """
        if ignore_index is not None and loss == "jaccard":
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )

        super().__init__()

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        num_classes: int = self.hparams["num_classes"]
        ignore_index = self.hparams["ignore_index"]
        if loss == "ce":
            if num_classes == 1:
                self.criterion = nn.BCEWithLogitsLoss(
                    weight=self.hparams["class_weights"]
                )
            else:
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
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        metrics = MetricCollection(
            [
                BinaryAccuracy()
                if num_classes == 1
                else MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                BinaryJaccardIndex()
                if num_classes == 1
                else MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="micro"
                ),
            ]
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
        weights: Optional[Union[WeightsEnum, str, bool]] = self.hparams["weights"]
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
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet'...."
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
        model: str = self.hparams["model"]
        image1 = batch["image1"]
        image2 = batch["image2"]
        y = batch["mask"].float()
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
            y_hat = self(x)
        elif model == "fcsiamdiff":
            x = torch.stack((image1, image2), dim=1)
            y_hat = self(x)
        else:
            raise ValueError(f"Model type '{model}' is not valid. ")
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss: Tensor = self.criterion(y_hat, y)
        loss: Tensor = self.criterion(y_hat, y)
        self.log("train_loss", loss)
        self.train_metrics(y_hat, y)
        self.log_dict(self.train_metrics)
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
        model: str = self.hparams["model"]
        image1 = batch["image1"]
        image2 = batch["image2"]
        y = batch["mask"].float()
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
            y_hat = self(x)
        elif model == "fcsiamdiff":
            x = torch.stack((image1, image2), dim=1)
            y_hat = self(x)
        else:
            raise ValueError(f"Model type '{model}' is not valid.")
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("val_loss", loss)
        self.val_metrics(y_hat, y)
        self.log_dict(self.val_metrics)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and hasattr(self.trainer.datamodule, "plot")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat.sigmoid().squeeze(1)
                for key in ["image1", "image2", "mask", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f"image/{batch_idx}", fig, global_step=self.global_step
                    )
                    plt.close()
            except ValueError:
                pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        model: str = self.hparams["model"]
        image1 = batch["image1"]
        image2 = batch["image2"]
        y = batch["mask"].float()
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
            y_hat = self(x)
        elif model == "fcsiamdiff":
            x = torch.stack((image1, image2), dim=1)
            y_hat = self(x)
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. " "Currently, only supports 'unet'"
            )
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)

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
        model: str = self.hparams["model"]
        image1 = batch["image1"]
        image2 = batch["image2"]
        if model == "unet":
            x = torch.cat([image1, image2], dim=1)
            y_hat: Tensor = self(x).softmax(dim=1)
            return y_hat
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. " "Currently, only supports 'unet'"
            )
