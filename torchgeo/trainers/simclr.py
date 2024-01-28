# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SimCLR trainer for self-supervised learning (SSL)."""

import os
import warnings
from typing import Any, Optional, Union

import kornia.augmentation as K
import lightning
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.models.modules import SimCLRProjectionHead
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models._api import WeightsEnum

import torchgeo.transforms as T

from ..models import get_weight
from . import utils
from .base import BaseTask


def simclr_augmentations(size: int, weights: Tensor) -> nn.Module:
    """Data augmentation used by SimCLR.

    Args:
        size: Size of patch to crop.
        weights: Weight vector for grayscale computation.

    Returns:
        Data augmentation pipeline.
    """
    # https://github.com/google-research/simclr/blob/master/data_util.py
    ks = size // 10 // 2 * 2 + 1
    return K.AugmentationSequential(
        K.RandomResizedCrop(size=(size, size), ratio=(0.75, 1.33)),
        K.RandomHorizontalFlip(),
        K.RandomVerticalFlip(),  # added
        # Not appropriate for multispectral imagery, seasonal contrast used instead
        # K.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8)
        K.RandomBrightness(brightness=(0.2, 1.8), p=0.8),
        K.RandomContrast(contrast=(0.2, 1.8), p=0.8),
        T.RandomGrayscale(weights=weights, p=0.2),
        K.RandomGaussianBlur(kernel_size=(ks, ks), sigma=(0.1, 2)),
        data_keys=["input"],
    )


class SimCLRTask(BaseTask):
    """SimCLR: a simple framework for contrastive learning of visual representations.

    Reference implementation:

    * https://github.com/google-research/simclr

    If you use this trainer in your research, please cite the following papers:

    * v1: https://arxiv.org/abs/2002.05709
    * v2: https://arxiv.org/abs/2006.10029

    .. versionadded:: 0.5
    """

    monitor = "train_loss"

    def __init__(
        self,
        model: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        version: int = 2,
        layers: int = 3,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        lr: float = 4.8,
        weight_decay: float = 1e-4,
        temperature: float = 0.07,
        memory_bank_size: int = 64000,
        gather_distributed: bool = False,
        size: int = 224,
        grayscale_weights: Optional[Tensor] = None,
        augmentations: Optional[nn.Module] = None,
    ) -> None:
        """Initialize a new SimCLRTask instance.

        Args:
            model: Name of the `timm
                <https://huggingface.co/docs/timm/reference/models>`__ model to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            version: Version of SimCLR, 1--2.
            layers: Number of layers in projection head (2 for v1, 3+ for v2).
            hidden_dim: Number of hidden dimensions in projection head
                (defaults to output dimension of model).
            output_dim: Number of output dimensions in projection head
                (defaults to output dimension of model).
            lr: Learning rate (0.3 x batch_size / 256 is recommended).
            weight_decay: Weight decay coefficient (1e-6 for v1, 1e-4 for v2).
            temperature: Temperature used in NT-Xent loss.
            memory_bank_size: Size of memory bank (0 for v1, 64K for v2).
            gather_distributed: Gather negatives from all GPUs during distributed
                training (ignored if memory_bank_size > 0).
            size: Size of patch to crop.
            grayscale_weights: Weight vector for grayscale computation, see
                :class:`~torchgeo.transforms.RandomGrayscale`. Only used when
                ``augmentations=None``. Defaults to average of all bands.
            augmentations: Data augmentation. Defaults to SimCLR augmentation.

        Raises:
            AssertionError: If an invalid version of SimCLR is requested.

        Warns:
            UserWarning: If hyperparameters do not match SimCLR version requested.
        """
        # Validate hyperparameters
        assert version in range(1, 3)
        if version == 1:
            if layers > 2:
                warnings.warn("SimCLR v1 only uses 2 layers in its projection head")
            if memory_bank_size > 0:
                warnings.warn("SimCLR v1 does not use a memory bank")
        elif version == 2:
            if layers == 2:
                warnings.warn("SimCLR v2 uses 3+ layers in its projection head")
            if memory_bank_size == 0:
                warnings.warn("SimCLR v2 uses a memory bank")

        self.weights = weights
        super().__init__(ignore=["weights", "augmentations"])

        grayscale_weights = grayscale_weights or torch.ones(in_channels)
        self.augmentations = augmentations or simclr_augmentations(
            size, grayscale_weights
        )

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        self.criterion = NTXentLoss(
            self.hparams["temperature"],
            self.hparams["memory_bank_size"],
            self.hparams["gather_distributed"],
        )

    def configure_models(self) -> None:
        """Initialize the model."""
        weights = self.weights
        hidden_dim: int = self.hparams["hidden_dim"]
        output_dim: int = self.hparams["output_dim"]

        # Create backbone
        self.backbone = timm.create_model(
            self.hparams["model"],
            in_chans=self.hparams["in_channels"],
            num_classes=0,
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
        input_dim = self.backbone.num_features
        if hidden_dim is None:
            hidden_dim = input_dim
        if output_dim is None:
            output_dim = input_dim

        self.projection_head = SimCLRProjectionHead(
            input_dim, hidden_dim, output_dim, self.hparams["layers"]
        )

        # Initialize moving average of output
        self.avg_output_std = 0.0

        # TODO
        # v1+: add global batch norm
        # v2: add selective kernels, channel-wise attention mechanism

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass of the model.

        Args:
            x: Mini-batch of images.

        Returns:
            Output of the model and backbone.
        """
        h: Tensor = self.backbone(x)  # shape of batch_size x num_features
        z = self.projection_head(h)
        return z, h

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

        Raises:
            AssertionError: If channel dimensions are incorrect.
        """
        x = batch["image"]

        in_channels: int = self.hparams["in_channels"]
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

        z1, h1 = self(x1)
        z2, h2 = self(x2)

        loss: Tensor = self.criterion(z1, z2)

        # Calculate the mean normalized standard deviation over features dimensions.
        # If this is << 1 / sqrt(h1.shape[1]), then the model is not learning anything.
        output = h1.detach()
        output = F.normalize(output, dim=1)
        output_std = torch.std(output, dim=0)
        output_std = torch.mean(output_std, dim=0)
        self.avg_output_std = 0.9 * self.avg_output_std + (1 - 0.9) * output_std.item()

        self.log("train_ssl_std", self.avg_output_std)
        self.log("train_loss", loss)

        return loss

    def validation_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op, does nothing."""

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""
        # TODO
        # v2: add distillation step

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""

    def configure_optimizers(
        self,
    ) -> "lightning.pytorch.utilities.types.OptimizerLRSchedulerConfig":
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        # Original paper uses LARS optimizer, but this is not defined in PyTorch
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )
        max_epochs = 200
        if self.trainer and self.trainer.max_epochs:
            max_epochs = self.trainer.max_epochs
        if self.hparams["version"] == 1:
            warmup_epochs = 10
        else:
            warmup_epochs = int(max_epochs * 0.05)
        scheduler = SequentialLR(
            optimizer,
            schedulers=[
                LinearLR(optimizer, total_iters=warmup_epochs),
                CosineAnnealingLR(optimizer, T_max=max_epochs),
            ],
            milestones=[warmup_epochs],
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": self.monitor},
        }
