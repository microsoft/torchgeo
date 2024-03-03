# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for semantic segmentation."""

import os
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch.nn as nn
from matplotlib.figure import Figure
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchvision.models._api import WeightsEnum

from ..datasets import RGBBandsMissingError, unbind_samples
from ..models import FCN, get_weight
from . import utils
from .base import BaseTask


class SemanticSegmentationTask(BaseTask):
    """Semantic Segmentation."""

    def __init__(
        self,
        model: str = "unet",
        backbone: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        num_filters: int = 3,
        loss: str = "ce",
        class_weights: Optional[Tensor] = None,
        ignore_index: Optional[int] = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """Initialize a new SemanticSegmentationTask instance.

        Args:
            model: Name of the
                `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
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
            num_filters: Number of filters. Only applicable when model='fcn'.
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

        .. versionchanged:: 0.3
           *ignore_zeros* was renamed to *ignore_index*.

        .. versionchanged:: 0.4
           *segmentation_model*, *encoder_name*, and *encoder_weights*
           were renamed to *model*, *backbone*, and *weights*.

        .. versionadded: 0.5
            The *class_weights*, *freeze_backbone*, and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           The *weights* parameter now supports WeightEnums and checkpoint paths.
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.

        .. versionchanged:: 0.6
            The *ignore_index* parameter now works for jaccard loss.
        """
        self.weights = weights
        super().__init__(ignore="weights")

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
        num_filters: int = self.hparams["num_filters"]

        if model == "unet":
            self.model = smp.Unet(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=backbone,
                encoder_weights="imagenet" if weights is True else None,
                in_channels=in_channels,
                classes=num_classes,
            )
        elif model == "fcn":
            self.model = FCN(
                in_channels=in_channels, classes=num_classes, num_filters=num_filters
            )
        else:
            raise ValueError(
                f"Model type '{model}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if model != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams["freeze_backbone"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams["freeze_decoder"] and model in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

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
            # JaccardLoss requires a list of classes to use instead of a class
            # index to ignore.
            classes = [
                i for i in range(self.hparams["num_classes"]) if i != ignore_index
            ]

            self.criterion = smp.losses.JaccardLoss(mode="multiclass", classes=classes)
        elif loss == "focal":
            self.criterion = smp.losses.FocalLoss(
                "multiclass", ignore_index=ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * Multiclass Pixel Accuracy: Ratio of correctly classified pixels.
          Uses 'micro' averaging. Higher values are better.
        * Multiclass Jaccard Index (IoU): Per-pixel overlap between predicted and
          actual segments. Uses 'macro' averaging. Higher values are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging, not used here, gives equal weight to each class, useful
             for balanced performance assessment across imbalanced classes.
        """
        num_classes: int = self.hparams["num_classes"]
        ignore_index: Optional[int] = self.hparams["ignore_index"]
        metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=num_classes,
                    ignore_index=ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                MulticlassJaccardIndex(
                    num_classes=num_classes, ignore_index=ignore_index, average="micro"
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

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
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
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
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
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
            datamodule = self.trainer.datamodule
            batch["prediction"] = y_hat.argmax(dim=1)
            for key in ["image", "mask", "prediction"]:
                batch[key] = batch[key].cpu()
            sample = unbind_samples(batch)[0]

            fig: Optional[Figure] = None
            try:
                fig = datamodule.plot(sample)
            except RGBBandsMissingError:
                pass

            if fig:
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self(x)
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
        x = batch["image"]
        y_hat: Tensor = self(x).softmax(dim=1)
        return y_hat
