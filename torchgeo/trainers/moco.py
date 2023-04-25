# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""MoCo trainer for self-supervised learning (SSL)."""

import os
import warnings
from typing import Optional, Union, cast
from collections.abc import Sequence

import kornia.augmentation as K
import timm
import torch
import torch.nn as nn
from lightly.loss import NTXentLoss
from lightly.models.modules.heads import ProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.utils.scheduler import cosine_schedule
from lightning import LightningModule
from torch import Tensor
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
)
from torchvision.models._api import WeightsEnum

from ..models import get_weight
from . import utils

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


# https://github.com/facebookresearch/moco/blob/main/main_moco.py#L326
# https://github.com/facebookresearch/moco-v3/blob/main/main_moco.py#L261
SIZE = 224
KS = SIZE // 10 // 2 * 2 + 1

# Same as InstDict: https://arxiv.org/abs/1805.01978
AUG1_V1 = AUG2_V1 = K.AugmentationSequential(
    K.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.2, 1)),
    # Not appropriate for multispectral imagery, seasonal contrast used instead
    # K.RandomGrayscale(p=0.2),
    # K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.4, p=1)
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),  # added
    data_keys=["input"],
)

# Similar to SimCLR: https://arxiv.org/abs/2002.05709
AUG1_V2 = AUG2_V2 = K.AugmentationSequential(
    K.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.2, 1)),
    # Not appropriate for multispectral imagery, seasonal contrast used instead
    # K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8)
    # K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur(kernel_size=(KS, KS), sigma=(0.1, 2), p=0.5),
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),  # added
    data_keys=["input"],
)

# Same as BYOL: https://arxiv.org/abs/2006.07733
AUG1_V3 = K.AugmentationSequential(
    K.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.08, 1)),
    # Not appropriate for multispectral imagery,
    # seasonal contrast used instead
    # K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8)
    # K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur(kernel_size=(KS, KS), sigma=(0.1, 2), p=1),
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),  # added
    data_keys=["input"],
)
AUG2_V3 = K.AugmentationSequential(
    K.RandomResizedCrop(size=(SIZE, SIZE), scale=(0.08, 1)),
    # Not appropriate for multispectral imagery,
    # seasonal contrast used instead
    # K.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8)
    # K.RandomGrayscale(p=0.2),
    K.RandomGaussianBlur(kernel_size=(KS, KS), sigma=(0.1, 2), p=0.1),
    K.RandomSolarize(p=0.2),
    K.RandomHorizontalFlip(),
    K.RandomVerticalFlip(),  # added
    data_keys=["input"],
)


# Remove once https://github.com/lightly-ai/lightly/pull/1163 is merged and released
class MoCoProjectionHead(ProjectionHead):
    """MoCo projection head."""

    def __init__(
        self,
        input_dim: int = 2048,
        hidden_dim: int = 4096,
        output_dim: int = 256,
        num_layers: int = 2,
        batch_norm: bool = True,
    ):
        """Initialize a new MoCoProjectionHead instance.

        Args:
            input_dim: Number of input dimensions.
            hidden_dim: Number of hidden dimensions.
            output_dim: Number of output dimensions.
            num_layers: Number of hidden layers.
            batch_norm: Whether or not to use batch norms.
        """
        layers: list[tuple[int, int, Optional[nn.Module], Optional[nn.Module]]] = []
        layers.append(
            (
                input_dim,
                hidden_dim,
                nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                nn.ReLU(),
            )
        )
        for _ in range(2, num_layers):
            layers.append(
                (
                    hidden_dim,
                    hidden_dim,
                    nn.BatchNorm1d(hidden_dim) if batch_norm else None,
                    nn.ReLU(),
                )
            )
        layers.append(
            (
                hidden_dim,
                output_dim,
                nn.BatchNorm1d(output_dim) if batch_norm else None,
                None,
            )
        )
        super().__init__(layers)


class MoCoTask(LightningModule):  # type: ignore[misc]
    """MoCo: Momentum Contrast.

    Reference implementations:

    * https://github.com/facebookresearch/moco
    * https://github.com/facebookresearch/moco-v3

    If you use this trainer in your research, please cite the following papers:

    * v1: https://arxiv.org/abs/1911.05722
    * v2: https://arxiv.org/abs/2003.04297
    * v3: https://arxiv.org/abs/2104.02057

    .. versionadded:: 0.5
    """

    def __init__(
        self,
        model: str = "resnet50",
        weights: Optional[Union[WeightsEnum, str, bool]] = None,
        in_channels: int = 3,
        version: int = 3,
        layers: int = 3,
        hidden_dim: int = 4096,
        output_dim: int = 256,
        lr: float = 9.6,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        schedule: Sequence[int] = [120, 160],
        temperature: float = 1,
        memory_bank_size: int = 0,
        moco_momentum: float = 0.99,
        gather_distributed: bool = False,
        augmentation1: nn.Module = AUG1_V3,
        augmentation2: nn.Module = AUG2_V3,
    ) -> None:
        """Initialize a new MoCoTask instance.

        Args:
            model: Name of the timm model to use.
            weights: Initial model weights. Either a weight enum, the string
                representation of a weight enum, True for ImageNet weights, False
                or None for random weights, or the path to a saved model state dict.
            in_channels: Number of input channels to model.
            version: Version of MoCo, 1--3.
            layers: Number of layers in projection head
                (not used in v1, 2 for v1/2, 3 for v3).
            hidden_dim: Number of hidden dimensions in projection head
                (not used in v1, 2048 for v2, 4096 for v3).
            output_dim: Number of output dimensions in projection head
                (not used in v1, 128 for v2, 256 for v3).
            lr: Learning rate (0.6 x batch_size / 256 is recommended).
            weight_decay: Weight decay coefficient (1e-4 for v1/2, 1e-6 for v3).
            momentum: Momentum of SGD solver (v1/2 only).
            schedule: Epochs at which to drop lr by 10x (v1/2 only).
            temperature: Temperature used in InfoNCE loss (0.07 for v1/2, 1 for v3).
            memory_bank_size: Size of memory bank (65536 for v1/2, 0 for v3).
            moco_momentum: MoCo momentum of updating key encoder
                (0.999 for v1/2, 0.99 for v3)
            gather_distributed: Gather negatives from all GPUs during distributed
                training (ignored if memory_bank_size > 0).
            augmentation1: Data augmentation for 1st branch.
            augmentation2: Data augmentation for 2nd branch.

        Raises:
            AssertionError: If an invalid version of MoCo is requested.

        Warns:
            UserWarning: If hyperparameters do not match MoCo version requested.
        """
        super().__init__()

        # Validate hyperparameters
        assert version in range(1, 4)
        if version == 1:
            if memory_bank_size == 0:
                warnings.warn("MoCo v1 uses a memory bank")
        elif version == 2:
            if layers > 2:
                warnings.warn("MoCo v2 only uses 2 layers in its projection head")
            if memory_bank_size == 0:
                warnings.warn("MoCo v2 uses a memory bank")
        elif version == 3:
            if layers == 2:
                warnings.warn("MoCo v3 uses 3 layers in its projection head")
            if memory_bank_size > 0:
                warnings.warn("MoCo v3 does not use a memory bank")

        self.save_hyperparameters(ignore=["augmentation1", "augmentation2"])
        self.augmentation1 = augmentation1
        self.augmentation2 = augmentation2

        # Create backbone
        self.backbone = timm.create_model(
            model, in_chans=in_channels, num_classes=0, pretrained=weights is True
        )
        self.backbone_momentum = timm.create_model(
            model, in_chans=in_channels, num_classes=0, pretrained=weights is True
        )
        deactivate_requires_grad(self.backbone_momentum)

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            elif os.path.exists(weights):
                _, state_dict = utils.extract_backbone(weights)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            self.backbone = utils.load_state_dict(self.backbone, state_dict)

        # Create projection (and prediction) head
        batch_norm = version == 3
        if version > 1:
            input_dim = self.backbone.num_features
            self.projection_head = MoCoProjectionHead(
                input_dim, hidden_dim, output_dim, layers, batch_norm=batch_norm
            )
            self.projection_head_momentum = MoCoProjectionHead(
                input_dim, hidden_dim, output_dim, layers, batch_norm=batch_norm
            )
            deactivate_requires_grad(self.projection_head_momentum)
        if version == 3:
            self.prediction_head = MoCoProjectionHead(
                output_dim, hidden_dim, output_dim, num_layers=2, batch_norm=batch_norm
            )

        # Define loss function
        self.criterion = NTXentLoss(temperature, memory_bank_size, gather_distributed)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            x: Mini-batch of images.

        Returns:
            Output from the model.
        """
        q = self.backbone(x)
        if self.hparams["version"] > 1:
            q = self.projection_head(q)
        if self.hparams["version"] == 3:
            q = self.prediction_head(q)
        return cast(Tensor, q)

    def forward_momentum(self, x: Tensor) -> Tensor:
        """Forward pass of the momentum model.

        Args:
            x: Mini-batch of images.

        Returns:
            Output from the momentum model.
        """
        k = self.backbone_momentum(x)
        if self.hparams["version"] > 1:
            k = self.projection_head_momentum(k)
        return cast(Tensor, k)

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
            x1 = self.augmentation1(x1)
            x2 = self.augmentation2(x2)

        m = self.hparams["moco_momentum"]
        if self.hparams["version"] == 1:
            q = self.forward(x1)
            with torch.no_grad():
                update_momentum(self.backbone, self.backbone_momentum, m)
                k = self.forward_momentum(x2)
            loss = self.criterion(q, k)
        elif self.hparams["version"] == 2:
            q = self.forward(x1)
            with torch.no_grad():
                update_momentum(self.backbone, self.backbone_momentum, m)
                update_momentum(self.projection_head, self.projection_head_momentum, m)
                k = self.forward_momentum(x2)
            loss = self.criterion(q, k)
        if self.hparams["version"] == 3:
            m = cosine_schedule(self.current_epoch, self.trainer.max_epochs, m, 1)
            q1 = self.forward(x1)
            q2 = self.forward(x2)
            with torch.no_grad():
                update_momentum(self.backbone, self.backbone_momentum, m)
                update_momentum(self.projection_head, self.projection_head_momentum, m)
                k1 = self.forward_momentum(x1)
                k2 = self.forward_momentum(x2)
            loss = self.criterion(q1, k2) + self.criterion(q2, k1)

        self.log("train_loss", loss)

        return cast(Tensor, loss)

    def validation_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""

    def test_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""

    def predict_step(self, batch: dict[str, Tensor], batch_idx: int) -> None:
        """No-op, does nothing."""

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        if self.hparams["version"] == 3:
            optimizer = AdamW(
                params=self.parameters(),
                lr=self.hparams["lr"],
                weight_decay=self.hparams["weight_decay"],
            )
            warmup_epochs = 40
            lr_scheduler = SequentialLR(
                optimizer,
                schedulers=[
                    LinearLR(
                        optimizer,
                        start_factor=1 / warmup_epochs,
                        total_iters=warmup_epochs,
                    ),
                    CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs),
                ],
                milestones=[warmup_epochs],
            )
        else:
            optimizer = SGD(
                params=self.parameters(),
                lr=self.hparams["lr"],
                momentum=self.hparams["momentum"],
                weight_decay=self.hparams["weight_decay"],
            )
            lr_scheduler = MultiStepLR(
                optimizer=optimizer, milestones=self.hparams["schedule"]
            )
        return [optimizer], [lr_scheduler]
