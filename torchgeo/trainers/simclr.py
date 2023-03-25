# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""SimCLR trainers for self-supervised learning (SSL)."""

from typing import Dict, List, Tuple

import kornia.augmentation as K
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR

try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import _LRScheduler as LRScheduler


class SimCLRTask(LightningModule):  # type: ignore[misc]
    """SimCLR: a simple framework for contrastive learning of visual representations.

    Implementation based on:

    * https://github.com/google-research/simclr
    * https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/13-contrastive-learning.html

    If you use this trainer in your research, please cite the following papers:

    * v1: https://arxiv.org/abs/2002.05709
    * v2: https://arxiv.org/abs/2006.10029

    .. versionadded:: 0.5
    """  # noqa: E501

    def __init__(
        self,
        model: str = "resnet50",
        in_channels: int = 3,
        version: int = 2,
        layers: int = 3,
        hidden_dim: int = 128,
        lr: float = 4.8,
        weight_decay: float = 1e-4,
        max_epochs: int = 100,
        temperature: float = 0.07,
    ) -> None:
        """Initialize a new SimCLRTask instance.

        Args:
            model: Name of the timm model to use.
            in_channels: Number of input channels to model.
            version: Version of SimCLR, 1--2.
            layers: Number of layers in projection head (2 for v1 or 3+ for v2).
            hidden_dim: Number of hidden dimensions in projection head.
            lr: Learning rate (0.3 x batch_size / 256 is recommended).
            weight_decay: Weight decay coefficient (1e-6 for v1 or 1e-4 for v2).
            max_epochs: Maximum number of epochs to train for.
            temperature: Temperature used in InfoNCE loss.
        """
        super().__init__()

        assert version in range(1, 3)

        self.save_hyperparameters()

        self.model = timm.create_model(
            model, in_chans=in_channels, num_classes=hidden_dim
        )

        # Projection head
        # https://github.com/google-research/simclr/blob/master/model_util.py#L141
        for i in range(layers):
            if i == layers - 1:
                # For the final layer, skip bias and ReLU
                self.model.fc = nn.Sequential(
                    self.model.fc, nn.Linear(hidden_dim, hidden_dim, bias=False)
                )
            else:
                # For the middle layers, use bias and ReLU
                self.model.fc = nn.Sequential(
                    self.model.fc,
                    nn.ReLU(inplace=True),
                    nn.Linear(hidden_dim, hidden_dim, bias=True),
                )

        # TODO
        # v1+: add global batch norm
        # v2: add selective kernels, channel-wise attention mechanism, memory bank

        # Data augmentation
        # https://github.com/google-research/simclr/blob/master/data_util.py
        self.aug = K.AugmentationSequential(
            K.RandomResizedCrop(size=(96, 96)),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),  # added
            # Not appropriate for multispectral imagery, seasonal contrast used instead
            # K.ColorJitter(
            #     brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2, p=0.8
            # )
            # K.RandomGrayscale(p=0.2),
            K.RandomGaussianBlur(kernel_size=9, sigma=(0.1, 2)),
            data_keys=["image"],
        )

    def forward(self, batch: Tensor) -> Tensor:
        """Forward pass of the model.

        Args:
            batch: Mini-batch of images.

        Returns:
            Output from the model.
        """
        batch = self.model(batch)
        return batch

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Tensor:
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

        # Apply augmentations independently for each season
        x1 = self.aug(x1)
        x2 = self.aug(x2)

        x = torch.cat([x1, x2], dim=0)

        # Encode all images
        x = self(x)

        # Calculate cosine similarity
        cos_sim = F.cosine_similarity(x[:, None, :], x[None, :, :], dim=-1)

        # Mask out cosine similarity to itself
        self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
        cos_sim.masked_fill_(self_mask, -9e15)

        # Find positive example -> batch_size // 2 away from the original example
        pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // 2, dims=0)

        # NT-Xent loss (aka InfoNCE loss)
        cos_sim = cos_sim / self.hparams["temperature"]
        nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
        nll = nll.mean()

        self.log("train_loss", nll)

        return nll

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
