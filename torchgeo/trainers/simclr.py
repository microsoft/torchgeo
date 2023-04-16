# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SimCLR trainers for self-supervised learning (SSL)."""

import os
from typing import Any, cast

import kornia.augmentation as K
import timm
import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

from ..models import get_weight
from . import utils

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


# https://github.com/google-research/simclr/blob/master/data_util.py
SIZE = 224
KS = SIZE // 10
AUG = K.AugmentationSequential(
    K.RandomResizedCrop(size=(SIZE, SIZE), ratio=(0.75, 1.33)),
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),  # added
    # Not appropriate for multispectral imagery, seasonal contrast used instead
    # K.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8)
    # K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur(kernel_size=(KS, KS), sigma=(0.1, 2)),
    data_keys=["input"],
)


class SimCLRTask(LightningModule):  # type: ignore[misc]
    """SimCLR: a simple framework for contrastive learning of visual representations.

    Reference implementation:

    * https://github.com/google-research/simclr

    If you use this trainer in your research, please cite the following papers:

    * v1: https://arxiv.org/abs/2002.05709
    * v2: https://arxiv.org/abs/2006.10029

    .. versionadded:: 0.5
    """

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
        self.hyperparams = cast(dict[str, Any], self.hparams)
        self.augmentations = kwargs.get("augmentations", AUG)
        self.config_model()

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

    def forward(self, batch: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            batch: Mini-batch of images.

        Returns:
            Output from the model.
        """
        h = self.backbone(batch)
        z = self.projection_head(h)
        return z

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.

        Returns:
            The loss tensor.
        """
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

        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""
        # TODO
        # v2: add distillation step

    def predict_step(self, batch: Dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[LRScheduler]]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        # Original paper uses LARS optimizer, but this is not defined in PyTorch
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.hparams["max_epochs"], eta_min=self.hparams["lr"] / 50
        )
        return [optimizer], [lr_scheduler]
