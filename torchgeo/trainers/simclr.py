# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SimCLR Trainer."""

import os
from typing import Any, Dict, Tuple, cast

import kornia.augmentation as K
import timm
import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightning import LightningModule
from torch import Tensor
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.models._api import WeightsEnum

from ..models import get_weight
from . import utils


def default_augmentations(image_size: Tuple[int, int] = (256, 256)):
    """Initialize a module for applying SimCLR augmentations.
    Args:
        image_size: Tuple of integers defining the image size
    Returns: augmentations
    """
    return nn.Sequential(
        K.Resize(size=image_size, align_corners=False),
        # Not suitable for multispectral adapt
        # K.ColorJitter(0.8, 0.8, 0.8, 0.8, 0.2),
        # K.RandomGrayscale(p=0.2),
        K.RandomHorizontalFlip(),
        K.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.1),
        K.RandomResizedCrop(size=image_size),
    )


class SimCLRTask(LightningModule):  # type: ignore[misc]
    """LightningModule for SimCLR pretraining.
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
        # Create backbone
        weights = self.hyperparams["weights"]
        self.backbone = timm.create_model(
            self.hyperparams["model"],
            num_classes=0,
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
            self.backbone = utils.load_state_dict(self.backbone, state_dict)

        # Create projection head
        self.projection_head = SimCLRProjectionHead(
            self.backbone.num_features,
            self.hyperparams["projection_hidden_size"],
            self.hyperparams["projection_output_size"],
        )

        # Define loss function
        self.loss = NTXentLoss(
            temperature=self.hyperparams["temperature"],
            gather_distributed=self.hyperparams["distributed"],
        )

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the LightningModule with a model.
        Args:
            model: Name of the encoder model use
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict.
            in_channels: Number of input channels to model
            projection_hidden_size: Number of hidden units in projection head
            projection_output_size: Output size of projection head
            temperature: NTXent loss function temperature
            distributed: Set True if using distributed training
            learning_rate: Learning rate for optimizer
            learning_rate_schedule_patience: Patience for learning rate scheduler
            augmentations: Augmentations to use for training
        .. versionchanged:: 0.5
        """
        super().__init__()

        # Creates `self.hparams` from kwargs
        self.save_hyperparameters(ignore=["augmentations"])
        self.hyperparams = cast(Dict[str, Any], self.hparams)
        self.augmentations = kwargs.get("augmentations", default_augmentations())
        self.config_model()

    def forward(self, x):
        """Forward pass of the model.
        Args:
            x: tensor of data to run through the model
        Returns:
            output from the model
        """
        h = self.backbone(x)
        z = self.projection_head(h)
        return z

    def training_step(self, *args: Any, **kwargs: Any) -> Tensor:
        """Compute and return the training loss.
        Args:
            batch: the output of your DataLoader
        Returns:
            training loss
        """
        batch = args[0]
        x = batch["image"]

        in_channels = self.hyperparams["in_channels"]
        assert x.size(1) == in_channels or x.size(1) == 2 * in_channels

        if x.size(1) == in_channels:
            x1 = x
            x2 = x
        else:
            x1 = x[:, :in_channels]
            x2 = x[:, in_channels:]

        with torch.no_grad():
            x1 = self.augmentations(x1)
            x2 = self.augmentations(x2)

        z1 = self(x1)
        z2 = self(x2)
        loss = self.loss(z1, z2)

        # by default, the train step logs every `log_every_n_steps` steps where
        # `log_every_n_steps` is a parameter to the `Trainer` object
        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return cast(Tensor, loss)

    def configure_optimizers(self) -> Dict[str, Any]:
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
                "monitor": "train_loss",
            },
        }
