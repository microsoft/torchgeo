# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""IJEPA trainer for self-supervised learning (SSL)."""

import copy

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia import augmentation as K
from lightly.data.collate import IJEPAMaskCollator
from lightly.models import utils
from lightly.models.modules import IJEPAPredictorTIMM
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from timm.models import VisionTransformer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models._api import WeightsEnum

from ..datasets.utils import Sample
from ..models import get_weight
from .base import BaseTask
from .utils import load_state_dict


def ijepa_augmentation(size: int = 224) -> K.AugmentationSequential:
    """Get the default IJEPA augmentation as a Kornia AugmentationSequential module."""
    return K.AugmentationSequential(
        K.RandomResizedCrop(
            size=(size, size),
            scale=(0.2, 1.0),
            ratio=(3 / 4, 4 / 3),
            resample='bicubic',
        ),
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=0.5),
        data_keys=['input'],
    )


class IJEPA(nn.Module):
    """IJEPA model for self-supervised learning.

    Consists of an online encoder, a target encoder (EMA update), and a predictor.
    The online encoder applies context masks before the transformer blocks, so
    only context patches attend to each other. The predictor predicts target
    patch features from the encoded context. The target encoder processes the
    full image (all patches attend to each other) to produce high-quality target
    representations for the loss.

    Reference implementations:

    * https://github.com/facebookresearch/ijepa
    * https://docs.lightly.ai/self-supervised-learning/examples/ijepa.html
    """

    def __init__(
        self, encoder: VisionTransformer, predictor: IJEPAPredictorTIMM
    ) -> None:
        """Initialize the IJEPA model.

        Args:
            encoder: A timm VisionTransformer model to use as the online encoder.
            predictor: An IJEPAPredictorTIMM module for predicting target features.
        """
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor
        self.target_encoder: VisionTransformer = copy.deepcopy(encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

    @staticmethod
    def _extract_patch_tokens(x: torch.Tensor, encoder: nn.Module) -> torch.Tensor:
        """Strip prefix tokens (CLS, registers) from the ViT output.

        Args:
            x: Output tensor from forward_features with shape
                (B, num_prefix + num_patches, D).
            encoder: The ViT encoder module.

        Returns:
            Tensor with only patch tokens, shape (B, num_patches, D).
        """
        num_prefix = getattr(encoder, 'num_prefix_tokens', 0)
        if num_prefix > 0:
            x = x[:, num_prefix:]
        return x

    def forward_target(
        self, images: torch.Tensor, masks_pred: list[torch.Tensor], nenc: int
    ) -> torch.Tensor:
        """Forward pass through the target encoder.

        Unlike the context encoder, the full image passes through all transformer
        blocks to produce high-quality target features. Masks are applied after
        the blocks to select target patch features. This matches the reference
        implementation.

        Args:
            images: Input images with shape (B, C, H, W).
            masks_pred: List of prediction mask tensors, each (B, num_pred_patches).
            nenc: Number of context masks (for repeat_interleave_batch).

        Returns:
            Target features with shape
                (B * nenc * npred, num_pred_patches, embed_dim).
        """
        with torch.no_grad():
            h = self.target_encoder.forward_features(images)
            h = self._extract_patch_tokens(h, self.target_encoder)
            h = F.layer_norm(h, (h.size(-1),))
            B = len(h)
            h = utils.apply_masks(h, masks_pred)
            h = utils.repeat_interleave_batch(h, B, repeat=nenc)
            return h

    def forward_context(
        self,
        images: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> torch.Tensor:
        """Forward pass through the online encoder and predictor.

        Masks are applied before the transformer blocks, matching the reference
        implementation. Only context patches pass through the attention layers,
        preventing information leakage from target regions.

        Args:
            images: Input images with shape (B, C, H, W).
            masks_enc: List of context mask tensors, each (B, num_context_patches).
            masks_pred: List of prediction mask tensors, each (B, num_pred_patches).

        Returns:
            Predicted features with shape
                (B * nenc * npred, num_pred_patches, embed_dim).
        """
        encoder = self.encoder
        x = encoder.patch_embed(images)
        x = encoder._pos_embed(x)
        x = self._extract_patch_tokens(x, encoder)
        x = utils.apply_masks(x, masks_enc)
        x = encoder.norm_pre(x)
        x = encoder.blocks(x)
        x = encoder.norm(x)
        z = self.predictor(x, masks_enc, masks_pred)
        return z

    def forward(
        self,
        images: torch.Tensor,
        masks_enc: list[torch.Tensor],
        masks_pred: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for both context and target branches.

        Args:
            images: Input images with shape (B, C, H, W).
            masks_enc: List of context mask tensors.
            masks_pred: List of prediction mask tensors.

        Returns:
            Tuple of (predicted_features, target_features) where predicted
            features are from context patches only (masked before transformer
            blocks) and target features are from target patches selected after
            a full-image forward pass.
        """
        nenc = len(masks_enc) if isinstance(masks_enc, list) else 1
        z = self.forward_context(images, masks_enc, masks_pred)
        h = self.forward_target(images, masks_pred, nenc)
        return z, h

    def update_target_encoder(self, momentum: float) -> None:
        """Update target encoder weights using EMA.

        Args:
            momentum: Momentum coefficient for the EMA update.
        """
        with torch.no_grad():
            for param_q, param_k in zip(
                self.encoder.parameters(), self.target_encoder.parameters()
            ):
                param_k.data.mul_(momentum).add_(
                    (1.0 - momentum) * param_q.detach().data
                )


class IJEPATask(BaseTask):
    """IJEPA: Joint-Embedding Predictive Architecture for self-supervised learning.

    Reference implementations:

    * https://github.com/facebookresearch/ijepa
    * https://docs.lightly.ai/self-supervised-learning/examples/ijepa.html

    If you use this code for your research, please cite the original paper:

    * https://arxiv.org/abs/2301.08243
    """

    ignore = ('transform', 'weights')
    monitor = 'train_loss'

    def __init__(
        self,
        model: str = 'vit_base_patch16_224',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        transform: nn.Module | None = None,
        predictor_embed_dim: int = 384,
        predictor_depth: int = 6,
        predictor_num_heads: int = 12,
        lr: float = 1e-3,
        weight_decay: float = 0.04,
        warmup_epochs: int = 40,
        size: int = 224,
        enc_mask_scale: tuple[float, float] = (0.85, 1.0),
        pred_mask_scale: tuple[float, float] = (0.15, 0.2),
        aspect_ratio: tuple[float, float] = (0.75, 1.5),
        nenc: int = 1,
        npred: int = 4,
        min_keep: int = 10,
        allow_overlap: bool = False,
        ema: tuple[float, float] = (0.996, 1.0),
    ) -> None:
        """Initialize the IJEPA task.

        Args:
            model: The ViT architecture to use for the encoder. Must be compatible
                with timm's create_model function.
            weights: Pretrained weights to initialize the encoder with. Can be a
                timm WeightsEnum or a string identifier for a timm weight, True to
                use default pretrained weights, or None for random initialization.
            in_channels: Number of input channels in the images. Must match the
                in_chans argument of the ViT model.
            transform: Optional transform to apply to input images. If None, a
                default augmentation (RandomResizedCrop + flips) is used.
            predictor_embed_dim: The inner embedding dimension of the predictor.
            predictor_depth: Number of transformer blocks in the predictor.
            predictor_num_heads: Number of attention heads in the predictor.
            lr: Learning rate for the optimizer.
            weight_decay: Weight decay for the AdamW optimizer.
            warmup_epochs: Number of linear warmup epochs before cosine annealing.
            size: The input image size (height and width) after augmentation.
            enc_mask_scale: Scale range for context (encoder) block masks.
            pred_mask_scale: Scale range for prediction (target) block masks.
            aspect_ratio: Aspect ratio range for prediction block masks.
            nenc: Number of context blocks per image.
            npred: Number of prediction blocks per image.
            min_keep: Minimum number of patches to keep in a mask block.
            allow_overlap: Whether to allow overlap between context and prediction
                masks.
            ema: EMA momentum range (start, end) for the target encoder update.
        """
        self.weights = weights
        super().__init__()
        self.transform = (
            transform if transform is not None else ijepa_augmentation(size)
        )
        self.warmup_epochs = warmup_epochs
        self.ema = ema
        self._momentum_iter: int = 0

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        self.criterion = nn.SmoothL1Loss()

    def configure_models(self) -> None:
        """Initialize the model."""
        model: str = self.hparams['model']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']

        try:
            vit = timm.create_model(
                model,
                in_chans=in_channels,
                num_classes=0,
                img_size=self.hparams['size'],
                pretrained=weights is True,
            )
        except Exception as e:
            raise ValueError('Model not compatible with IJEPA:', e)
        if not isinstance(vit, VisionTransformer):
            raise ValueError('Model not compatible with IJEPA:', vit.__class__.__name__)


        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            load_state_dict(vit, state_dict)

        self.patch_size = vit.patch_embed.patch_size[0]
        num_patches = vit.patch_embed.num_patches
        embed_dim = vit.embed_dim

        predictor = IJEPAPredictorTIMM(
            num_patches=num_patches,
            depth=self.hparams['predictor_depth'],
            mlp_dim=embed_dim,
            predictor_embed_dim=self.hparams['predictor_embed_dim'],
            num_heads=self.hparams['predictor_num_heads'],
        )

        self.model: IJEPA = IJEPA(encoder=vit, predictor=predictor)

        self.mask_collator: IJEPAMaskCollator = IJEPAMaskCollator(
            input_size=self.hparams['size'],
            patch_size=self.patch_size,
            enc_mask_scale=self.hparams['enc_mask_scale'],
            pred_mask_scale=self.hparams['pred_mask_scale'],
            aspect_ratio=self.hparams['aspect_ratio'],
            nenc=self.hparams['nenc'],
            npred=self.hparams['npred'],
            min_keep=self.hparams['min_keep'],
            allow_overlap=self.hparams['allow_overlap'],
        )

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Initialize the optimizer and learning rate scheduler.

        Returns:
            Optimizer and learning rate scheduler.
        """
        optim = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams['lr'],
            weight_decay=self.hparams['weight_decay'],
            betas=(0.9, 0.95),
        )
        max_epochs = 300
        if self.trainer and self.trainer.max_epochs is not None:
            max_epochs = self.trainer.max_epochs
        warmup_epochs = min(self.warmup_epochs, max_epochs)
        warmup = LinearLR(optim, 1e-8, 1, total_iters=warmup_epochs)
        if max_epochs > warmup_epochs:
            cosine = CosineAnnealingLR(
                optim, T_max=max_epochs - warmup_epochs, eta_min=0
            )
            scheduler = SequentialLR(
                optim, [warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            scheduler = warmup

        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    def _get_momentum(self) -> float:
        """Get the current EMA momentum value.

        Linearly increases from ema[0] to ema[1] over the course of training.

        Returns:
            The momentum value for the current training step.
        """
        self._momentum_iter += 1
        max_steps = 1
        try:
            max_steps = self.trainer.estimated_stepping_batches  # type: ignore[union-attr]
        except RuntimeError:
            pass
        progress = self._momentum_iter / max(max_steps, 1)
        progress = min(progress, 1.0)
        momentum: float = self.ema[0] + progress * (self.ema[1] - self.ema[0])
        return momentum

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the training loss and update the target encoder.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        with torch.no_grad():
            images = self.transform(batch['image'].float())

        batch_size = images.shape[0]
        device = images.device

        dummy_batch = [torch.zeros(1) for _ in range(batch_size)]
        _, masks_enc, masks_pred = self.mask_collator(dummy_batch)
        masks_enc = [m.to(device) for m in masks_enc]
        masks_pred = [m.to(device) for m in masks_pred]

        z, h = self.model(images, masks_enc, masks_pred)
        loss = self.criterion(z, h)

        momentum = self._get_momentum()
        self.model.update_target_encoder(momentum)

        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        self.log(
            'momentum', momentum, on_step=True, on_epoch=False, batch_size=batch_size
        )

        return loss

    def validation_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op, does nothing."""

    def test_step(self, batch: Sample, batch_idx: int, dataloader_idx: int = 0) -> None:
        """No-op, does nothing."""

    def predict_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> None:
        """No-op, does nothing."""
