# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SimCLR trainers for self-supervised learning (SSL)."""

import os
from typing import Optional, Union

import kornia.augmentation as K
import timm
import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import ProjectionHead
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models._api import WeightsEnum

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


# Lightly implementation doesn't support SimCLR v2
# TODO: upstream our implementation
class SimCLRProjectionHead(ProjectionHead):
    """SimCLR projection head."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 3,
    ) -> None:
        """Initialize a new SimCLRProjectionHead instance.

        Args:
            input_dim: Number of input dimensions.
            hidden_dim: Number of hidden dimensions.
            output_dim: Number of output dimensions.
            num_layers: Number of hidden layers.
        """
        super()(
            [
                (input_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU()),
                (hidden_dim, hidden_dim, nn.BatchNorm1d(hidden_dim), nn.ReLU())
                * (num_layers - 2),
                (hidden_dim, output_dim, nn.BatchNorm1d(output_dim), None),
            ]
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

    def __init__(
        self,
        model: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        version: int = 2,
        layers: int = 3,
        hidden_dim: int = 128,
        output_dim: int = 128,
        lr: float = 4.8,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        temperature: float = 0.07,
        distributed: bool = False,
        augmentations: nn.Module = AUG,
    ) -> None:
        """Initialize a new SimCLRTask instance.

        Args:
            model: Name of the timm model to use.
            weights: Either a weight enum, the string representation of a weight enum,
                True for ImageNet weights, False or None for random weights,
                or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            version: Version of SimCLR, 1--2.
            layers: Number of layers in projection head (2 for v1 or 3+ for v2).
            hidden_dim: Number of hidden dimensions in projection head.
            output_dim: Number of output dimensions in projection head.
            lr: Learning rate (0.3 x batch_size / 256 is recommended).
            weight_decay: Weight decay coefficient (1e-6 for v1 or 1e-4 for v2).
            max_epochs: Maximum number of epochs to train for.
            temperature: Temperature used in InfoNCE loss.
            distributed: Use distributed training.
            augmentations: Data augmentation.
        """
        super().__init__()

        self.save_hyperparameters()

        # Create backbone
        weights = self.hparams["weights"]
        self.backbone = timm.create_model(
            self.hparams["model"],
            num_classes=self.hparams["hidden_dim"],
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
            self.backbone = utils.load_state_dict(self.backbone, state_dict)

        # Create projection head
        self.projection_head = SimCLRProjectionHead(
            self.backbone.num_features,
            self.hparams["hidden_dim"],
            self.hparams["output_dim"],
            self.hparams["layers"],
        )

        # Define loss function
        self.criterion = NTXentLoss(
            temperature=self.hparams["temperature"],
            gather_distributed=self.hparams["distributed"],
        )

        # TODO
        # v1+: add global batch norm
        # v2: add selective kernels, channel-wise attention mechanism, memory bank

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

    def training_step(self, batch: dict[str, Tensor], batch_idx: int) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.

        Returns:
            The loss tensor.
        """
        x = batch["image"]

        in_channels = self.hparams["in_channels"]
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

        loss = self.criterion(z1, z2)

        self.log("train_loss", loss, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""
        # TODO
        # v2: add distillation step

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
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
