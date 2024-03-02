# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for image classification."""

import os
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassFBetaScore,
    MulticlassJaccardIndex,
    MultilabelAccuracy,
    MultilabelFBetaScore,
)
from torchvision.models._api import WeightsEnum

from ..datasets import unbind_samples
from ..models import get_weight
from . import utils
from .base import BaseTask


class ClassificationTask(BaseTask):
    """Image classification."""

    def __init__(
        self,
        model: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        num_classes: int = 1000,
        loss: str = "ce",
        class_weights: Optional[Tensor] = None,
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
    ) -> None:
        """Initialize a new ClassificationTask instance.

        Args:
            model: Name of the `timm
                <https://huggingface.co/docs/timm/reference/models>`__ model to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            num_classes: Number of prediction classes.
            loss: One of 'ce', 'bce', 'jaccard', or 'focal'.
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to linear probe
                the classifier head.

        .. versionchanged:: 0.4
           *classification_model* was renamed to *model*.

        .. versionadded:: 0.5
           The *class_weights* and *freeze_backbone* parameters.

        .. versionchanged:: 0.5
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.
        """
        self.weights = weights
        super().__init__(ignore="weights")

    def configure_losses(self) -> None:
        """Initialize the loss criterion.

        Raises:
            ValueError: If *loss* is invalid.
        """
        loss: str = self.hparams["loss"]
        if loss == "ce":
            self.criterion: nn.Module = nn.CrossEntropyLoss(
                weight=self.hparams["class_weights"]
            )
        elif loss == "bce":
            self.criterion = nn.BCEWithLogitsLoss()
        elif loss == "jaccard":
            self.criterion = JaccardLoss(mode="multiclass")
        elif loss == "focal":
            self.criterion = FocalLoss(mode="multiclass", normalized=True)
        else:
            raise ValueError(f"Loss type '{loss}' is not valid.")

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * Multiclass Overall Accuracy (OA): Ratio of correctly classified pixels.
          Uses 'micro' averaging. Higher values are better.
        * Multiclass Average Accuracy (AA): Ratio of correctly classified classes.
          Uses 'macro' averaging. Higher values are better.
        * Multiclass Jaccard Index (IoU): Per-class overlap between predicted and
          actual classes. Uses 'macro' averaging. Higher valuers are better.
        * Multiclass F1 Score: The harmonic mean of precision and recall.
          Uses 'micro' averaging. Higher values are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not reflect
             minority class accuracy.
           * 'Macro' averaging gives equal weight to each class, and is useful for
             balanced performance assessment across imbalanced classes.
        """
        metrics = MetricCollection(
            {
                "OverallAccuracy": MulticlassAccuracy(
                    num_classes=self.hparams["num_classes"], average="micro"
                ),
                "AverageAccuracy": MulticlassAccuracy(
                    num_classes=self.hparams["num_classes"], average="macro"
                ),
                "JaccardIndex": MulticlassJaccardIndex(
                    num_classes=self.hparams["num_classes"]
                ),
                "F1Score": MulticlassFBetaScore(
                    num_classes=self.hparams["num_classes"], beta=1.0, average="micro"
                ),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights

        # Create model
        self.model = timm.create_model(
            self.hparams["model"],
            num_classes=self.hparams["num_classes"],
            in_chans=self.hparams["in_channels"],
            pretrained=weights is True,
        )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            utils.load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hparams["freeze_backbone"]:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

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
        y = batch["label"]
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
        y = batch["label"]
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
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat.argmax(dim=-1)
                for key in ["image", "label", "prediction"]:
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
        x = batch["image"]
        y = batch["label"]
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
        y_hat: Tensor = self(x).softmax(dim=-1)
        return y_hat


class MultiLabelClassificationTask(ClassificationTask):
    """Multi-label image classification."""

    def configure_metrics(self) -> None:
        """Initialize the performance metrics.

        * Multiclass Overall Accuracy (OA): Ratio of correctly classified pixels.
          Uses 'micro' averaging. Higher values are better.
        * Multiclass Average Accuracy (AA): Ratio of correctly classified classes.
          Uses 'macro' averaging. Higher values are better.
        * Multiclass F1 Score: The harmonic mean of precision and recall.
          Uses 'micro' averaging. Higher values are better.

        .. note::
           * 'Micro' averaging suits overall performance evaluation but may not
             reflect minority class accuracy.
           * 'Macro' averaging gives equal weight to each class, and is useful for
             balanced performance assessment across imbalanced classes.
        """
        metrics = MetricCollection(
            {
                "OverallAccuracy": MultilabelAccuracy(
                    num_labels=self.hparams["num_classes"], average="micro"
                ),
                "AverageAccuracy": MultilabelAccuracy(
                    num_labels=self.hparams["num_classes"], average="macro"
                ),
                "F1Score": MultilabelFBetaScore(
                    num_labels=self.hparams["num_classes"], beta=1.0, average="micro"
                ),
            }
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
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = torch.sigmoid(y_hat)
        loss: Tensor = self.criterion(y_hat, y.to(torch.float))
        self.log("train_loss", loss)
        self.train_metrics(y_hat_hard, y)
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
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = torch.sigmoid(y_hat)
        loss = self.criterion(y_hat, y.to(torch.float))
        self.log("val_loss", loss)
        self.val_metrics(y_hat_hard, y)
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
                batch["prediction"] = y_hat_hard
                for key in ["image", "label", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                if fig:
                    summary_writer = self.logger.experiment
                    summary_writer.add_figure(
                        f"image/{batch_idx}", fig, global_step=self.global_step
                    )
            except ValueError:
                pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """Compute the test loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = torch.sigmoid(y_hat)
        loss = self.criterion(y_hat, y.to(torch.float))
        self.log("test_loss", loss)
        self.test_metrics(y_hat_hard, y)
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
        y_hat = torch.sigmoid(self(x))
        return y_hat
