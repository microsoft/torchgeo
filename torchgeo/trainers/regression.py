# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tasks."""

from typing import Any, Dict, cast

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from packaging.version import parse
from torch import Tensor
from torch.nn.modules import Conv2d, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from ..datasets.utils import unbind_samples

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Conv2d.__module__ = "nn.Conv2d"
Linear.__module__ = "nn.Linear"


class RegressionTask(pl.LightningModule):
    """LightningModule for training models on regression datasets."""

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters."""
        model = self.hyperparams["model"]
        pretrained = self.hyperparams["pretrained"]

        if parse(torchvision.__version__) >= parse("0.13"):
            if pretrained:
                kwargs = {
                    "weights": getattr(
                        torchvision.models, f"ResNet{model[6:]}_Weights"
                    ).DEFAULT
                }
            else:
                kwargs = {"weights": None}
        else:
            kwargs = {"pretrained": pretrained}

        self.model = getattr(torchvision.models, model)(**kwargs)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, out_features=1)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LightningModule for training simple regression models.

        Keyword Args:
            model: Name of the model to use
            learning_rate: Initial learning rate to use in the optimizer
            learning_rate_schedule_patience: Patience parameter for the LR scheduler
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()  # type: ignore[operator]
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.config_task()

        self.train_metrics = MetricCollection(
            {"RMSE": MeanSquaredError(squared=False), "MAE": MeanAbsoluteError()},
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass of the model.

        Args:
            x: tensor of data to run through the model

        Returns:
            output from the model
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

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute validation loss and log example predictions.

        Args:
            batch: the output of your DataLoader
            batch_idx: the index of this batch
        """
        batch = args[0]
        batch_idx = args[1]
        x = batch["image"]
        y = batch["label"].view(-1, 1)
        y_hat = self.forward(x)

        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        self.val_metrics(y_hat, y)

        if batch_idx < 10:
            try:
                datamodule = self.trainer.datamodule  # type: ignore[attr-defined]
                batch["prediction"] = y_hat
                for key in ["image", "label", "prediction"]:
                    batch[key] = batch[key].cpu()
                sample = unbind_samples(batch)[0]
                fig = datamodule.plot(sample)
                summary_writer = self.logger.experiment  # type: ignore[union-attr]
                summary_writer.add_figure(
                    f"image/{batch_idx}", fig, global_step=self.global_step
                )
            except AttributeError:
                pass

    def validation_epoch_end(self, outputs: Any) -> None:
        """Logs epoch level validation metrics.

        Args:
            outputs: list of items returned by validation_step
        """
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        """Compute test loss.

        Args:
            batch: the output of your DataLoader
        """
        batch = args[0]
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
