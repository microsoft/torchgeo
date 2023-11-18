# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for regression."""

import os
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torchvision.models._api import WeightsEnum

from ..datasets import unbind_samples
from ..models import FCN, get_weight
from . import utils
from .base import BaseTask


class RegressionTask(BaseTask):
    """Regression."""

    target_key = "label"

    def __init__(
        self,
        model: str = "resnet50",
        backbone: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        num_outputs: int = 1,
        num_filters: int = 3,
        loss: str = "mse",
        lr: float = 1e-3,
        patience: int = 10,
        freeze_backbone: bool = False,
        freeze_decoder: bool = False,
    ) -> None:
        """Initialize a new RegressionTask instance.

        Args:
            model: Name of the
                `timm <https://huggingface.co/docs/timm/reference/models>`__ or
                `smp <https://smp.readthedocs.io/en/latest/models.html>`__ model to use.
            backbone: Name of the
                `timm <https://smp.readthedocs.io/en/latest/encoders_timm.html>`__ or
                `smp <https://smp.readthedocs.io/en/latest/encoders.html>`__ backbone
                to use. Only applicable to PixelwiseRegressionTask.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            num_outputs: Number of prediction outputs.
            num_filters: Number of filters. Only applicable when model='fcn'.
            loss: One of 'mse' or 'mae'.
            lr: Learning rate for optimizer.
            patience: Patience for learning rate scheduler.
            freeze_backbone: Freeze the backbone network to linear probe
                the regression head. Does not support FCN models.
            freeze_decoder: Freeze the decoder network to linear probe
                the regression head. Does not support FCN models.
                Only applicable to PixelwiseRegressionTask.

        .. versionchanged:: 0.4
           Change regression model support from torchvision.models to timm

        .. versionadded:: 0.5
           The *freeze_backbone* and *freeze_decoder* parameters.

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
        if loss == "mse":
            self.criterion: nn.Module = nn.MSELoss()
        elif loss == "mae":
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'mse' or 'mae' loss."
            )

    def configure_metrics(self) -> None:
        """Initialize the performance metrics."""
        metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def configure_models(self) -> None:
        """Initialize the model."""
        # Create model
        weights = self.weights
        self.model = timm.create_model(
            self.hparams["model"],
            num_classes=self.hparams["num_outputs"],
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
            self.model = utils.load_state_dict(self.model, state_dict)

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
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
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
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
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
                if self.target_key == "mask":
                    y = y.squeeze(dim=1)
                    y_hat = y_hat.squeeze(dim=1)
                batch["prediction"] = y_hat
                for key in ["image", self.target_key, "prediction"]:
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
        # TODO: remove .to(...) once we have a real pixelwise regression dataset
        y = batch[self.target_key].to(torch.float)
        y_hat = self(x)
        if y_hat.ndim != y.ndim:
            y = y.unsqueeze(dim=1)
        loss = self.criterion(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)
        self.log_dict(self.test_metrics)

    def predict_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the predicted regression values.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            Output predicted probabilities.
        """
        x = batch["image"]
        y_hat: Tensor = self(x)
        return y_hat


class PixelwiseRegressionTask(RegressionTask):
    """LightningModule for pixelwise regression of images.

    .. versionadded:: 0.5
    """

    target_key = "mask"

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights

        if self.hparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hparams["in_channels"],
                classes=1,
            )
        elif self.hparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hparams["in_channels"],
                classes=1,
            )
        elif self.hparams["model"] == "fcn":
            self.model = FCN(
                in_channels=self.hparams["in_channels"],
                classes=1,
                num_filters=self.hparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hparams['model']}' is not valid. "
                "Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if self.hparams["model"] != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hparams.get("freeze_backbone", False) and self.hparams["model"] in [
            "unet",
            "deeplabv3+",
        ]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hparams.get("freeze_decoder", False) and self.hparams["model"] in [
            "unet",
            "deeplabv3+",
        ]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False
