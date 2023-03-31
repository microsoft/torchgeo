# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Classification tasks."""

import os
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
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


class ClassificationTask(LightningModule):  # type: ignore[misc]
    """LightningModule for image classification.

    Supports any available `Timm model
    <https://huggingface.co/docs/timm/index>`_
    as an architecture choice. To see a list of available
    models, you can do:

    .. code-block:: python

        import timm
        print(timm.list_models())
    """

    def config_model(self) -> None:
        """Configures the model based on kwargs parameters passed to the constructor."""
        # Create model
        weights = self.hyperparams["weights"]
        self.model = timm.create_model(
            self.hyperparams["model"],
            num_classes=self.hyperparams["num_classes"],
            in_chans=self.hyperparams["in_channels"],
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
            self.model = utils.load_state_dict(self.model, state_dict)

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.config_model()

        if self.hyperparams["loss"] == "ce":
            self.loss: nn.Module = nn.CrossEntropyLoss()
        elif self.hyperparams["loss"] == "jaccard":
            self.loss = JaccardLoss(mode="multiclass")
        elif self.hyperparams["loss"] == "focal":
            self.loss = FocalLoss(mode="multiclass", normalized=True)
        else:
            raise ValueError(f"Loss type '{self.hyperparams['loss']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            model: Name of the classification model use
            loss: Name of the loss function, accepts 'ce', 'jaccard', or 'focal'
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict.
            num_classes: Number of prediction classes
            in_channels: Number of input channels to model
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        .. versionchanged:: 0.4
           The *classification_model* parameter was renamed to *model*.
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
        self.hyperparams = cast(Dict[str, Any], self.hparams)

        self.config_task()

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": MulticlassAccuracy(
                    num_classes=self.hyperparams["num_classes"], average="micro"
                ),
                "AverageAccuracy": MulticlassAccuracy(
                    num_classes=self.hyperparams["num_classes"], average="macro"
                ),
                "JaccardIndex": MulticlassJaccardIndex(
                    num_classes=self.hyperparams["num_classes"]
                ),
                "F1Score": MulticlassFBetaScore(
                    num_classes=self.hyperparams["num_classes"],
                    beta=1.0,
                    average="micro",
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: input image

        Returns:
            prediction
        """
        return self.model(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def on_train_epoch_end(self) -> None:
        """Logs epoch-level training metrics."""
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
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
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
                plt.close()
            except ValueError:
                pass

    def on_validation_epoch_end(self) -> None:
        """Logs epoch level validation metrics."""
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader

        Returns:
            predicted softmax probabilities
        """
        batch = args[0]
        x = batch["image"]
        y_hat: Tensor = self(x).softmax(dim=-1)
        return y_hat

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            learning rate dictionary
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.hyperparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer,
                    patience=self.hyperparams["learning_rate_schedule_patience"],
                ),
                "monitor": "val_loss",
            },
        }


class MultiLabelClassificationTask(ClassificationTask):
    """LightningModule for multi-label image classification."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.config_model()

        if self.hyperparams["loss"] == "bce":
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Loss type '{self.hyperparams['loss']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            model: Name of the classification model use
            loss: Name of the loss function, currently only supports 'bce'
            weights: Either "random" or 'imagenet'
            num_classes: Number of prediction classes
            in_channels: Number of input channels to model
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        .. versionchanged:: 0.4
           The *classification_model* parameter was renamed to *model*.
        """
        super().__init__(**kwargs)

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": MultilabelAccuracy(
                    num_labels=self.hyperparams["num_classes"], average="micro"
                ),
                "AverageAccuracy": MultilabelAccuracy(
                    num_labels=self.hyperparams["num_classes"], average="macro"
                ),
                "F1Score": MultilabelFBetaScore(
                    num_labels=self.hyperparams["num_classes"],
                    beta=1.0,
                    average="micro",
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.

        Args:
            batch: the output of your DataLoader

        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y.to(torch.float))

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y.to(torch.float))

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
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
                summary_writer = self.logger.experiment
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
            except ValueError:
                pass

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
        x = batch["image"]
        y = batch["label"]
        y_hat = self(x)
        y_hat_hard = torch.sigmoid(y_hat)

        loss = self.loss(y_hat, y.to(torch.float))

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader
        Returns:
            predicted sigmoid probabilities
        """
        batch = args[0]
        x = batch["image"]
        y_hat = torch.sigmoid(self(x))
        return y_hat
