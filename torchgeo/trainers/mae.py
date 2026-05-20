# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""MAE trainer for self-supervised learning (SSL)."""

import timm
import torch
from kornia import augmentation as K
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from timm.models import VisionTransformer
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torchvision.models._api import WeightsEnum

from ..datasets.utils import Sample
from ..models import get_weight
from .base import BaseTask
from .utils import load_state_dict


def mae_augmentation(size: int = 224) -> K.AugmentationSequential:
    """Get the default MAE augmentations as a Kornia AugmentationSequential module."""
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


class MAETask(BaseTask):
    """MAE: Masked Autoencoder for self-supervised learning.

    Reference implementations:

    * https://github.com/facebookresearch/mae
    * https://docs.lightly.ai/self-supervised-learning/examples/mae.html

    If you use this code for your research, please cite the original paper:

    * https://arxiv.org/abs/2111.06377
    """

    ignore = ('transform', 'weights')

    def __init__(
        self,
        model: str = 'vit_base_patch32_224',
        weights: WeightsEnum | str | bool | None = None,
        in_channels: int = 3,
        transform: nn.Module | None = None,
        decoder_dim: int = 512,
        lr: float = 1.5e-4,
        decoder_num_heads: int = 8,
        decoder_depth: int = 1,
        weight_decay: float = 0.05,
        mask_ratio: float = 0.75,
        size: int = 224,
        norm_pix_loss: bool = True,
        warmup_epochs: int = 40,
    ) -> None:
        """Initialize the MAE task.

        Args:
            model: The ViT architecture to use for the encoder. Must be compatible with
                timm's create_model function.
            weights: Pretrained weights to initialize the encoder with. Can be a timm
                WeightsEnum or a string identifier for a timm weight, True to use
                default pretrained weights, or None for random initialization.
            in_channels: Number of input channels in the images. Must match the in_chans
                argument of the ViT model.
            transform: Optional transform to apply to the input images. If None, a
                default MAE augmentation will be used.
            decoder_dim: The embedding dimension of the MAE decoder. Typically 512 is a
                good choice for ViT-Base encoders.
            lr: Should typically be set to 1.5e-4 * batch_size / 256.
            decoder_num_heads: Number of attention heads in the MAE decoder.
            decoder_depth: Number of layers in the MAE decoder. Typically 1-4 layers is
                sufficient for good performance.
            weight_decay: Weight decay for the AdamW optimizer.
            mask_ratio: The ratio of tokens to mask during training. Typically 0.75 is a
                good choice.
            size: The input image size (height and width) after augmentation. Must match
                the input size expected by the ViT model.
            norm_pix_loss: If True, normalize each target patch to zero mean and unit
                variance before computing MSE. Recommended by the original MAE paper.
            warmup_epochs: Number of linear warmup epochs before cosine annealing.
        """
        self.weights = weights
        super().__init__()
        self.transform = transform if transform is not None else mae_augmentation(size)
        self.warmup_epochs = warmup_epochs

    def configure_losses(self) -> None:
        """Initialize the loss criterion."""
        self.criterion = nn.MSELoss(reduction='none')

    def configure_models(self) -> None:
        """Initialize the model."""
        model: str = self.hparams['model']
        weights = self.weights
        in_channels: int = self.hparams['in_channels']

        vit = timm.create_model(
            model, in_chans=in_channels, num_classes=0, pretrained=weights is True
        )
        if not isinstance(vit, VisionTransformer):
            raise ValueError(
                f'Model {model} is not a ViT architecture, which is required for MAE training.'
            )

        # Load weights
        if weights and weights is not True:
            if isinstance(weights, WeightsEnum):
                state_dict = weights.get_state_dict(progress=True)
            else:
                state_dict = get_weight(weights).get_state_dict(progress=True)
            load_state_dict(vit, state_dict)  # type: ignore[invalid-argument-type]

        self.patch_size = vit.patch_embed.patch_size[0]
        self.num_prefix_tokens = vit.num_prefix_tokens

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=self.hparams['decoder_dim'],
            decoder_depth=self.hparams['decoder_depth'],
            decoder_num_heads=self.hparams['decoder_num_heads'],
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            in_chans=vit.patch_embed.proj.in_channels,
        )

        self.sequence_length = self.backbone.sequence_length

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
        max_epochs = 800
        if self.trainer and self.trainer.max_epochs is not None:
            max_epochs = self.trainer.max_epochs
        # Ensure warmup does not exceed total number of epochs.
        warmup_epochs = min(self.warmup_epochs, max_epochs)
        # Cosine annealing with linear warmup.
        warmup = LinearLR(optim, 1e-8, 1, total_iters=warmup_epochs)
        if max_epochs > warmup_epochs:
            cosine_T_max = max(1, max_epochs - warmup_epochs)
            cosine = CosineAnnealingLR(optim, T_max=cosine_T_max, eta_min=0)
            scheduler = SequentialLR(
                optim, [warmup, cosine], milestones=[warmup_epochs]
            )
        else:
            # If training for fewer epochs than the warmup, only use the warmup schedule.
            scheduler = warmup

        return {
            'optimizer': optim,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }

    def forward(
        self, images: torch.Tensor, idx_keep: torch.Tensor, idx_mask: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through MAE encoder and decoder.

        Args:
            images: The input images, with shape (B, in_channels, H, W).
            idx_keep: The indices of the tokens that were kept (not masked), with shape
                (B, N_keep).
            idx_mask: The indices of the tokens that were masked, with shape (B,
                N_mask).

        Returns:
            The predicted pixel values for the masked tokens, with shape (B, N_mask,
                patch_size*patch_size*in_channels).
        """
        # Encode the visible tokens with the MAE encoder
        x_encoded = self.backbone.encode(images=images, idx_keep=idx_keep)
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))
        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)
        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(
        self, batch: Sample, batch_idx: int, dataloader_idx: int = 0
    ) -> torch.Tensor:
        """Compute the training loss and additional metrics.

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
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.hparams['mask_ratio'],
            device=images.device,
        )
        x_pred = self.forward(images, idx_keep, idx_mask)
        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - self.num_prefix_tokens)
        # Apply normalization to target patches if norm_pix_loss is True.
        if self.hparams['norm_pix_loss']:
            target = utils.normalize_mean_var(target, dim=-1)
        # per-sample loss for std logging
        loss_per_sample = self.criterion(x_pred, target)
        loss_per_sample = loss_per_sample.mean(dim=list(range(1, loss_per_sample.ndim)))
        loss = loss_per_sample.mean()

        psnr = -10.0 * torch.log10(loss.detach().clamp(min=1e-10))

        self.log('train_loss', loss, on_step=True, on_epoch=True, batch_size=batch_size)
        # Near-zero std indicates that the model is learning uniformly across samples.
        self.log(
            'train_loss_std',
            loss_per_sample.std(),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        # Near-zero means that the model is predicting a constant value, which is a common failure mode for MAE training.
        self.log(
            'pred_std',
            x_pred.std(),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        # If this is very low, the model is getting "easy" patches (smooth background) and the loss won't be meaningful.
        self.log(
            'target_std',
            target.std(),
            on_step=True,
            on_epoch=False,
            batch_size=batch_size,
        )
        # PSNR is a common metric for image reconstruction quality. Higher is better, and values above ~30 indicate good reconstruction.
        self.log('psnr', psnr, on_step=False, on_epoch=True, batch_size=batch_size)

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
