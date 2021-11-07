# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tasks."""

from typing import Any, Dict

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Conv2d, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torchvision import models

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Conv2d.__module__ = "nn.Conv2d"
Linear.__module__ = "nn.Linear"


class RegressionTask(pl.LightningModule):
    """LightningModule for training models on regression datasets."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters."""
        if self.hparams["model"] == "resnet18":
            self.model = models.resnet18(pretrained=True)
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(  # type: ignore[attr-defined]
                in_features, out_features=1
            )
        else:
            raise ValueError(f"Model type '{self.hparams['model']}' is not valid.")

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LightningModule for training simple regression models.

        Keyword Args:
            model: Name of the model to use
            learning_rate: Initial learning rate to use in the optimizer
            learning_rate_schedule_patience: Patience parameter for the LR scheduler
        """
        super().__init__()
        self.save_hyperparameters()  # creates `self.hparams` from kwargs
        self.config_task()

        self.train_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, x: Tensor) -> Any:  # type: ignore[override]
        """Forward pass of the model."""
        return self.model(x)

    def training_step(  # type: ignore[override]
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Tensor:
        """Training step with an MSE loss.

        Args:
            batch: Current batch
            batch_idx: Index of current batch

        Returns:
            training loss
        """
        x = batch["image"]
        y = batch["label"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss)  # logging to TensorBoard
        self.train_metrics(y_hat, y)

        return loss

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
        y = batch["label"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        self.val_metrics(y_hat, y)

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
        y = batch["label"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)

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
