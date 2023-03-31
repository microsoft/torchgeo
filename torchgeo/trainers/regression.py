# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tasks."""

import os
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import timm
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torchvision.models._api import WeightsEnum

from ..datasets import unbind_samples
from ..models import get_weight
from . import utils


class RegressionTask(LightningModule):  # type: ignore[misc]
    """LightningModule for training models on regression datasets.

    Supports any available `Timm model
    <https://huggingface.co/docs/timm/index>`_
    as an architecture choice. To see a list of available
    models, you can do:

    .. code-block:: python

        import timm
        print(timm.list_models())
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters."""
        # Create model
        weights = self.hyperparams["weights"]
        self.model = timm.create_model(
            self.hyperparams["model"],
            num_classes=self.hyperparams["num_outputs"],
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

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LightningModule for training simple regression models.

        Keyword Args:
            model: Name of the timm model to use
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict.
            num_outputs: Number of prediction outputs
            in_channels: Number of input channels to model
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        .. versionchanged:: 0.4
            Change regression model support from torchvision.models to timm
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
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
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)

        self.log("train_loss", loss)  # logging to TensorBoard
        self.train_metrics(y_hat, y)

        return loss

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
        y = batch["label"].view(-1, 1)
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)
        self.val_metrics(y_hat, y)

        if (
            batch_idx < 10
            and hasattr(self.trainer, "datamodule")
            and self.logger
            and hasattr(self.logger, "experiment")
            and hasattr(self.logger.experiment, "add_figure")
        ):
            try:
                datamodule = self.trainer.datamodule
                batch["prediction"] = y_hat
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
        y = batch["label"].view(-1, 1)
        y_hat = self(x)

        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)

    def on_test_epoch_end(self) -> None:
        """Logs epoch level test metrics."""
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def predict_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the predictions.

        Args:
            batch: the output of your DataLoader
        Returns:
            predicted values
        """
        batch = args[0]
        x = batch["image"]
        y_hat: Tensor = self(x)
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
