# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Classification tasks."""

import os
from typing import Any, Dict, cast

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import FocalLoss, JaccardLoss
from torch import Tensor
from torch.nn.modules import Conv2d, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import Accuracy, FBeta, IoU, MetricCollection

from . import utils

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Conv2d.__module__ = "nn.Conv2d"
Linear.__module__ = "nn.Linear"


class ClassificationTask(pl.LightningModule):
    """LightningModule for image classification."""

    def config_model(self) -> None:
        """Configures the model based on kwargs parameters passed to the constructor."""
        in_channels = self.hparams["in_channels"]
        classification_model = self.hparams["classification_model"]

        imagenet_pretrained = False
        custom_pretrained = False
        if self.hparams["weights"] and not os.path.exists(self.hparams["weights"]):
            if self.hparams["weights"] == "imagenet":
                imagenet_pretrained = True
            elif self.hparams["weights"] == "random":
                imagenet_pretrained = False
            else:
                raise ValueError(
                    f"Weight type '{self.hparams['weights']}' is not valid."
                )
            custom_pretrained = False
        else:
            custom_pretrained = True

        # Create the model
        valid_models = timm.list_models(pretrained=True)
        if classification_model in valid_models:
            self.model = timm.create_model(
                classification_model,
                num_classes=self.hparams["num_classes"],
                in_chans=in_channels,
                pretrained=imagenet_pretrained,
            )
        else:
            raise ValueError(
                f"Model type '{classification_model}' is not a valid timm model."
            )

        if custom_pretrained:
            name, state_dict = utils.extract_encoder(self.hparams["weights"])

            if self.hparams["classification_model"] != name:
                raise ValueError(
                    f"Trying to load {name} weights into a "
                    f"{self.hparams['classification_model']}"
                )
            self.model = utils.load_state_dict(self.model, state_dict)

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.config_model()

        if self.hparams["loss"] == "ce":
            self.loss = nn.CrossEntropyLoss()  # type: ignore[attr-defined]
        elif self.hparams["loss"] == "jaccard":
            self.loss = JaccardLoss(mode="multiclass")
        elif self.hparams["loss"] == "focal":
            self.loss = FocalLoss(mode="multiclass", normalized=True)
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            classification_model: Name of the classification model use
            loss: Name of the loss function
            weights: Either "random", "imagenet_only", "imagenet_and_random", or
                "random_rgb"
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    num_classes=self.hparams["num_classes"], average="micro"
                ),
                "AverageAccuracy": Accuracy(
                    num_classes=self.hparams["num_classes"], average="macro"
                ),
                "IoU": IoU(num_classes=self.hparams["num_classes"]),
                "F1Score": FBeta(
                    num_classes=self.hparams["num_classes"], beta=1.0, average="micro"
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model.

        Args:
            x: input image

        Returns:
            prediction
        """
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self.forward(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def training_epoch_end(self, outputs: Any) -> None:
        """Logs epoch-level training metrics.

        Args:
            outputs: list of items returned by training_step
        """
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["label"]
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
        """Test step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["label"]
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
        optimizer = torch.optim.AdamW(
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


class MultiLabelClassificationTask(ClassificationTask):
    """LightningModule for multi-label image classification."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        self.config_model()

        if self.hparams["loss"] == "bce":
            self.loss = nn.BCEWithLogitsLoss()  # type: ignore[attr-defined]
        else:
            raise ValueError(f"Loss type '{self.hparams['loss']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            classification_model: Name of the classification model use
            loss: Name of the loss function
            weights: Either "random", "imagenet_only", "imagenet_and_random", or
                "random_rgb"
        """
        super().__init__(**kwargs)
        self.save_hyperparameters()  # creates `self.hparams` from kwargs

        self.config_task()

        self.train_metrics = MetricCollection(
            {
                "OverallAccuracy": Accuracy(
                    num_classes=self.hparams["num_classes"],
                    average="micro",
                    multiclass=False,
                ),
                "AverageAccuracy": Accuracy(
                    num_classes=self.hparams["num_classes"],
                    average="macro",
                    multiclass=False,
                ),
                "F1Score": FBeta(
                    num_classes=self.hparams["num_classes"],
                    beta=1.0,
                    average="micro",
                    multiclass=False,
                ),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self.forward(x)
        y_hat_hard = torch.softmax(y_hat, dim=-1)  # type: ignore[attr-defined]

        loss = self.loss(y_hat, y.to(torch.float))  # type: ignore[attr-defined]

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def validation_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Validation step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self.forward(x)
        y_hat_hard = torch.softmax(y_hat, dim=-1)  # type: ignore[attr-defined]

        loss = self.loss(y_hat, y.to(torch.float))  # type: ignore[attr-defined]

        self.log("val_loss", loss, on_step=False, on_epoch=True)
        self.val_metrics(y_hat_hard, y)

    def test_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> None:
        """Test step.

        Args:
            batch: Current batch
            batch_idx: Index of current batch
        """
        x = batch["image"]
        y = batch["label"]
        y_hat = self.forward(x)
        y_hat_hard = torch.softmax(y_hat, dim=-1)  # type: ignore[attr-defined]

        loss = self.loss(y_hat, y.to(torch.float))  # type: ignore[attr-defined]

        # by default, the test and validation steps only log per *epoch*
        self.log("test_loss", loss, on_step=False, on_epoch=True)
        self.test_metrics(y_hat_hard, y)
