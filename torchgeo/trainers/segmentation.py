# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Segmentation tasks."""

import os
import warnings
from typing import Any, cast

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassAccuracy, MulticlassJaccardIndex
from torchvision.models._api import WeightsEnum

from ..datasets.utils import unbind_samples
from ..models import FCN, get_weight
from . import utils


class SemanticSegmentationTask(LightningModule):
    """LightningModule for semantic segmentation of images.

    Supports `Segmentation Models Pytorch
    <https://github.com/qubvel/segmentation_models.pytorch>`_
    as an architecture choice in combination with any of these
    `TIMM backbones <https://smp.readthedocs.io/en/latest/encoders_timm.html>`_.
    """

    def config_task(self) -> None:
        """Configures the task based on kwargs parameters passed to the constructor."""
        weights = self.hyperparams["weights"]

        if self.hyperparams["model"] == "unet":
            self.model = smp.Unet(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["model"] == "deeplabv3+":
            self.model = smp.DeepLabV3Plus(
                encoder_name=self.hyperparams["backbone"],
                encoder_weights="imagenet" if weights is True else None,
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
            )
        elif self.hyperparams["model"] == "fcn":
            self.model = FCN(
                in_channels=self.hyperparams["in_channels"],
                classes=self.hyperparams["num_classes"],
                num_filters=self.hyperparams["num_filters"],
            )
        else:
            raise ValueError(
                f"Model type '{self.hyperparams['model']}' is not valid. "
                f"Currently, only supports 'unet', 'deeplabv3+' and 'fcn'."
            )

        if self.hyperparams["loss"] == "ce":
            ignore_value = -1000 if self.ignore_index is None else self.ignore_index

            class_weights = None
            if isinstance(self.class_weights, torch.Tensor):
                class_weights = self.class_weights.to(dtype=torch.float32)
            elif hasattr(self.class_weights, "__array__") or self.class_weights:
                class_weights = torch.tensor(self.class_weights, dtype=torch.float32)

            self.loss = nn.CrossEntropyLoss(
                ignore_index=ignore_value, weight=class_weights
            )
        elif self.hyperparams["loss"] == "jaccard":
            self.loss = smp.losses.JaccardLoss(
                mode="multiclass", classes=self.hyperparams["num_classes"]
            )
        elif self.hyperparams["loss"] == "focal":
            self.loss = smp.losses.FocalLoss(
                "multiclass", ignore_index=self.ignore_index, normalized=True
            )
        else:
            raise ValueError(
                f"Loss type '{self.hyperparams['loss']}' is not valid. "
                f"Currently, supports 'ce', 'jaccard' or 'focal' loss."
            )

        if self.hyperparams["model"] != "fcn":
            if weights and weights is not True:
                if isinstance(weights, WeightsEnum):
                    state_dict = weights.get_state_dict(progress=True)
                elif os.path.exists(weights):
                    _, state_dict = utils.extract_backbone(weights)
                else:
                    state_dict = get_weight(weights).get_state_dict(progress=True)
                self.model.encoder.load_state_dict(state_dict)

        # Freeze backbone
        if self.hyperparams.get("freeze_backbone", False) and self.hyperparams[
            "model"
        ] in ["unet", "deeplabv3+"]:
            for param in self.model.encoder.parameters():
                param.requires_grad = False

        # Freeze decoder
        if self.hyperparams.get("freeze_decoder", False) and self.hyperparams[
            "model"
        ] in ["unet", "deeplabv3+"]:
            for param in self.model.decoder.parameters():
                param.requires_grad = False

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model and loss function.

        Keyword Args:
            model: Name of the segmentation model type to use
            backbone: Name of the timm backbone to use
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict. FCN model does not support
                pretrained weights. Pretrained ViT weight enums are not supported yet.
            in_channels: Number of channels in input image
            num_classes: Number of semantic classes to predict
            loss: Name of the loss function, currently supports
                'ce', 'jaccard' or 'focal' loss
            class_weights: Optional rescaling weight given to each
                class and used with 'ce' loss
            ignore_index: Optional integer class index to ignore in the loss and metrics
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler
            freeze_backbone: Freeze the backbone network to fine-tune the
                decoder and segmentation head
            freeze_decoder: Freeze the decoder network to linear probe
                the segmentation head

        Raises:
            ValueError: if kwargs arguments are invalid

        .. versionchanged:: 0.3
           The *ignore_zeros* parameter was renamed to *ignore_index*.

        .. versionchanged:: 0.4
           The *segmentation_model* parameter was renamed to *model*,
           *encoder_name* renamed to *backbone*, and
           *encoder_weights* to *weights*.

        .. versionadded: 0.5
            The *class_weights*, *freeze_backbone*,
            and *freeze_decoder* parameters.

        .. versionchanged:: 0.5
           The *weights* parameter now supports WeightEnums and checkpoint paths.

        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters()
        self.hyperparams = cast(dict[str, Any], self.hparams)

        if not isinstance(kwargs["ignore_index"], (int, type(None))):
            raise ValueError("ignore_index must be an int or None")
        if (kwargs["ignore_index"] is not None) and (kwargs["loss"] == "jaccard"):
            warnings.warn(
                "ignore_index has no effect on training when loss='jaccard'",
                UserWarning,
            )
        self.ignore_index = kwargs["ignore_index"]
        self.class_weights = kwargs.get("class_weights", None)

        self.config_task()

        self.train_metrics = MetricCollection(
            [
                MulticlassAccuracy(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    multidim_average="global",
                    average="micro",
                ),
                MulticlassJaccardIndex(
                    num_classes=self.hyperparams["num_classes"],
                    ignore_index=self.ignore_index,
                    average="micro",
                ),
            ],
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
        y = batch["mask"]
        y_hat = self(x)
        y_hat_hard = y_hat.argmax(dim=1)

        loss = self.loss(y_hat, y)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)
        self.train_metrics(y_hat_hard, y)

        return cast(Tensor, loss)

    def on_train_epoch_end(self) -> None:
        """Logs epoch level training metrics."""
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
        y = batch["mask"]
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
                for key in ["image", "mask", "prediction"]:
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
        y = batch["mask"]
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

        By default, this will loop over images in a dataloader and aggregate
        predictions into a list. This may not be desirable if you have many images
        or large images which could cause out of memory errors. In this case
        it's recommended to override this with a custom predict_step.

        Args:
            batch: the output of your DataLoader

        Returns:
            predicted softmax probabilities
        """
        batch = args[0]
        x = batch["image"]
        y_hat: Tensor = self(x).softmax(dim=1)
        return y_hat

    def configure_optimizers(self) -> dict[str, Any]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            learning rate dictionary
        """
        optimizer = torch.optim.Adam(
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
