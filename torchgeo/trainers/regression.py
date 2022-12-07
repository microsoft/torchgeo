# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Regression tasks."""

import os
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import timm
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules import Conv2d, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection

from ..datasets.utils import unbind_samples
from . import utils

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Conv2d.__module__ = "nn.Conv2d"
Linear.__module__ = "nn.Linear"


class RegressionTask(pl.LightningModule):
    """LightningModule for training models on regression datasets.

    Supports any available `Timm model
    <https://rwightman.github.io/pytorch-image-models/>`_
    as an architecture choice. To see a list of available
    models, you can do:

    .. code-block:: python

        import timm
        print(timm.list_models())
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters."""
        in_channels = self.hyperparams["in_channels"]
        model = self.hyperparams["model"]

        imagenet_pretrained = False
        custom_pretrained = False
        if self.hyperparams["weights"] and not os.path.exists(
            self.hyperparams["weights"]
        ):
            if self.hyperparams["weights"] not in ["imagenet", "random"]:
                raise ValueError(
                    f"Weight type '{self.hyperparams['weights']}' is not valid."
                )
            else:
                imagenet_pretrained = self.hyperparams["weights"] == "imagenet"
            custom_pretrained = False
        else:
            custom_pretrained = True

        # Create the model
        valid_models = timm.list_models(pretrained=imagenet_pretrained)
        if model in valid_models:
            self.model = timm.create_model(
                model,
                num_classes=self.hyperparams["num_outputs"],
                in_chans=in_channels,
                pretrained=imagenet_pretrained,
            )
        else:
            raise ValueError(f"Model type '{model}' is not a valid timm model.")

        if custom_pretrained:
            name, state_dict = utils.extract_encoder(self.hyperparams["weights"])

            if self.hyperparams["model"] != name:
                raise ValueError(
                    f"Trying to load {name} weights into a "
                    f"{self.hyperparams['model']}"
                )
            self.model = utils.load_state_dict(self.model, state_dict)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize a new LightningModule for training simple regression models.

        Keyword Args:
            model: Name of the timm model to use
            weights: Either "random" or "imagenet"
            num_outputs: Number of prediction outputs
            in_channels: Number of input channels to model
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler

        .. versionchanged:: 0.4
            Change regression model support from torchvision.models to timm
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
        y_hat = self(x)

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
        y_hat = self(x)

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
                plt.close()
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
        y_hat = self(x)

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
