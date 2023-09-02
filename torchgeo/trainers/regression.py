# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Trainers for regression."""

import os
from typing import Any, Optional, Union, cast

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import timm
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MetricCollection
from torchvision.models._api import WeightsEnum

from ..datasets import unbind_samples
from ..models import FCN, get_weight
from .utils import extract_backbone, load_state_dict


class RegressionTask(LightningModule):
    """Regression."""

    target_key: str = "label"

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
            model: Name of the timm or segmentation_models_pytorch model to use.
            backbone: Name of the timm model to use.
                Only applicable to PixelwiseRegressionTask.
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

        Raises:
            ValueError: If any arguments are invalid.

        .. versionadded:: 0.5
           The *freeze_backbone* and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           *learning_rate* and *learning_rate_schedule_patience* were renamed to
           *lr* and *patience*.

        .. versionchanged:: 0.4
           Change regression model support from torchvision.models to timm
        """
        super().__init__()

        self.save_hyperparameters()
        self._configure_models()

        self.loss: nn.Module
        if loss == "mse":
            self.loss = nn.MSELoss()
        elif loss == "mae":
            self.loss = nn.L1Loss()
        else:
            raise ValueError(
                f"Loss type '{loss}' is not valid. "
                "Currently, supports 'mse' or 'mae' loss."
            )

        self.train_metrics = MetricCollection(
            {
                "RMSE": MeanSquaredError(squared=False),
                "MSE": MeanSquaredError(squared=True),
                "MAE": MeanAbsoluteError(),
            },
            prefix="train_",
        )
        self.val_metrics = self.train_metrics.clone(prefix="val_")
        self.test_metrics = self.train_metrics.clone(prefix="test_")

    def _configure_models(self) -> None:
        """Initialize the model."""
        # Create model
        weights = self.hparams["weights"]
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
                _, state_dict = extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            self.model = load_state_dict(self.model, state_dict)

        # Freeze backbone and unfreeze classifier head
        if self.hparams["freeze_backbone"]:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.get_classifier().parameters():
                param.requires_grad = True

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Mini-batch of images.

        Returns:
            Output from the model.
        """
        z = self.model(x)
        return cast(Tensor, z)

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

        loss: Tensor = self.loss(y_hat, y)
        self.log("train_loss", loss)
        self.train_metrics(y_hat, y)

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

        loss = self.loss(y_hat, y)
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
                if self.target_key == "mask":
                    y = y.squeeze(dim=1)
                    y_hat = y_hat.squeeze(dim=1)
                batch["prediction"] = y_hat
                for key in ["image", self.target_key, "prediction"]:
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

        loss = self.loss(y_hat, y)
        self.log("test_loss", loss)
        self.test_metrics(y_hat, y)

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
        y_hat: Tensor = self(x)
        return y_hat

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optimizer = AdamW(self.parameters(), lr=self.hparams["lr"])
        scheduler = ReduceLROnPlateau(optimizer, patience=self.hparams["patience"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


class PixelwiseRegressionTask(RegressionTask):
    """LightningModule for pixelwise regression of images.

    Supports `Segmentation Models Pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice in combination with any of these
    `TIMM backbones <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_.

    .. versionadded:: 0.5
    """

    target_key: str = "mask"

    def _configure_models(self) -> None:
        """Initialize the model."""
        weights = self.hparams["weights"]

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
                    _, state_dict = extract_backbone(weights)
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
