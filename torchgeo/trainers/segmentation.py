# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Segmentation tasks."""

from typing import Any, Dict, cast

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from pytorch_lightning.core.lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, IoU, MetricCollection

from ..models import FCN

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
DataLoader.__module__ = "torch.utils.data"


class SemanticSegmentationTask(LightningModule):
    """LightningModule for semantic segmentation of images."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        if self.hparams["segmentation_model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["encoder_name"],
                encoder_weights=self.hparams["encoder_weights"],
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
            )
        elif self.hparams["segmentation_model"] == "fcn":
            self.model = FCN(
                in_channels=self.hparams["in_channels"],
                classes=self.hparams["num_classes"],
                num_filters=self.hparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['segmentation_model']}' is not valid."
            )

        if self.hparams["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss(  # type: ignore[attr-defined]
                ignore_index=0
            )
        elif self.hparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hparams["num_classes"]
            )
        elif self.hparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=0, normalized=True
            )
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            segmentation_model: Name of the segmentation model type to use
            encoder_name: Name of the encoder model backbone to use
            encoder_weights: None or "imagenet" to use imagenet pretrained weights in
                the encoder model
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            loss: Name of the loss function

        Raises:
            ValueError: if kwargs arguments are invalid
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

        self.train_metrics = MetricCollection(
            [
                Accuracy(num_classes=self.hparams["num_classes"], ignore_index=0),
                IoU(num_classes=self.hparams["num_classes"], ignore_index=0),
            ],
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
        """
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step - reports average accuracy and average IoU.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step - reports average accuracy and average IoU.

        Logs the first 10 validation samples to tensorboard as images with 3 subplots
        showing the image, mask, and predictions.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step identical to the validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["mask"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)

    def test_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level test metrics.

        Args:
            outputs: list of items returned by test_step
        """
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self) -> Dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            a "lr dict" according to the pytorch lightning documentation --
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.hparams["learning_rate"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(
                    optimizer, patience=self.hparams["learning_rate_schedule_patience"]
                ),
                "monitor": "val_loss",
            },
        }
