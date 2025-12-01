# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""ChangeViT model implementation.

Based on the paper: https://arxiv.org/pdf/2406.12847
"""

from collections.abc import Sequence
from typing import Any

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torch.nn.modules import Module


class DetailCaptureModule(Module):
    """Detail capture module using timm's ResNet18 implementation.

    Paper states: 'three residual convolutional blocks (C2-C4) adapted from ResNet18'
    that generate 'three-scale detailed features: 1/2, 1/4, and 1/8 resolutions'
    with 'channel dimensions of FCi are set to 64, 128, and 256, respectively.'

    Uses timm's pretrained ResNet18 with projection layers to match paper specifications.
    """

    def __init__(
        self, in_channels: int = 6, backbone: str = 'resnet18', pretrained: bool = False
    ) -> None:
        """Initialize the detail capture module.

        Args:
            in_channels: Number of input channels (typically 6 for bitemporal RGB).
            backbone: Name of the timm backbone model to use.
            pretrained: Whether to load pretrained weights from timm.
        """
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=[0, 1, 2],
            in_chans=in_channels,
        )

        backbone_channels: list[int] = self.backbone.feature_info.channels()  # type: ignore[union-attr, operator]

        self.proj1 = nn.Conv2d(backbone_channels[0], 64, kernel_size=1)
        self.proj2 = nn.Conv2d(backbone_channels[1], 128, kernel_size=1)
        self.proj3 = nn.Conv2d(backbone_channels[2], 256, kernel_size=1)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass through detail capture module.

        Args:
            x: Bitemporal input tensor [B, 2*C, H, W]

        Returns:
            Tuple of features at 1/2, 1/4, and 1/8 scales with 64, 128, 256 channels
        """
        features = self.backbone(x)

        c2 = self.proj1(features[0])
        c3 = self.proj2(features[1])
        c4 = self.proj3(features[2])

        return c2, c3, c4


class FeatureInjector(Module):
    """Feature injector using cross-attention to inject detail features into ViT.

    Implements the cross-attention mechanism described in the ChangeViT paper,
    where ViT features serve as queries and detail features as keys/values.
    """

    def __init__(
        self,
        vit_dim: int,
        detail_dims: Sequence[int] = (64, 128, 256),
        num_heads: int = 8,
    ) -> None:
        """Initialize the feature injector.

        Args:
            vit_dim: Dimension of ViT features
            detail_dims: Dimensions of detail features at 3 scales (C2, C3, C4)
            num_heads: Number of attention heads
        """
        super().__init__()

        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=vit_dim, num_heads=num_heads, batch_first=True
                )
                for _ in range(3)
            ]
        )

        self.detail_projs = nn.ModuleList(
            [nn.Linear(dim, vit_dim) for dim in detail_dims]
        )

        self.fusion = nn.Sequential(
            nn.Linear(vit_dim * 4, vit_dim),
            nn.ReLU(inplace=True),
            nn.Linear(vit_dim, vit_dim),
        )

    def forward(
        self, vit_feats: Tensor, detail_feats: tuple[Tensor, Tensor, Tensor]
    ) -> Tensor:
        """Inject detail features into ViT features via cross-attention.

        Args:
            vit_feats: ViT features [B, N, D] where N = H*W/patch_size^2
            detail_feats: Tuple of detail features at 3 scales

        Returns:
            Enhanced ViT features [B, N, D]
        """
        _b, n, _d = vit_feats.shape
        enhanced_feats = [vit_feats]

        patch_grid_size = int(n**0.5)
        target_spatial = (patch_grid_size, patch_grid_size)

        for i, (detail_feat, cross_attn, proj) in enumerate(
            zip(detail_feats, self.cross_attns, self.detail_projs)
        ):
            detail_aligned = F.adaptive_avg_pool2d(detail_feat, target_spatial)
            detail_flat = detail_aligned.flatten(2).transpose(1, 2)
            detail_proj = proj(detail_flat)

            enhanced_feat, _ = cross_attn(
                query=vit_feats, key=detail_proj, value=detail_proj
            )
            enhanced_feats.append(enhanced_feat)

        fused = torch.cat(enhanced_feats, dim=-1)
        result: Tensor = self.fusion(fused)
        return result


class ChangeViTDecoder(Module):
    """Change detection decoder for ViT-based models.

    As described in the ChangeViT paper, this decoder handles the final difference
    modeling and change map generation from enhanced ViT features.
    """

    def __init__(
        self,
        in_channels: int = 768,
        inner_channels: int = 64,
        num_convs: int = 3,
        num_classes: int = 1,
    ) -> None:
        """Initialize the ChangeViTDecoder.

        Args:
            in_channels: Input feature dimension (ViT embedding dim)
            inner_channels: Number of inner channels for processing
            num_convs: Number of convolutional layers
            num_classes: Number of output classes
        """
        super().__init__()

        layers: list[nn.Module] = [
            nn.Sequential(
                nn.Conv2d(in_channels * 2, inner_channels, 3, 1, 1),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(True),
            )
        ]

        layers.extend(
            [
                nn.Sequential(
                    nn.Conv2d(inner_channels, inner_channels, 3, 1, 1),
                    nn.BatchNorm2d(inner_channels),
                    nn.ReLU(True),
                )
                for _ in range(num_convs - 1)
            ]
        )

        self.convs = nn.Sequential(*layers)
        self.head = nn.Conv2d(inner_channels, num_classes, 3, 1, 1)

    def forward(self, bi_feature: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for change detection.

        Args:
            bi_feature: Bitemporal features [B, T, C, H, W]

        Returns:
            Tuple of bidirectional change predictions (logits)
        """
        batch_size = bi_feature.size(0)

        t1t2 = torch.cat([bi_feature[:, 0], bi_feature[:, 1]], dim=1)
        t2t1 = torch.cat([bi_feature[:, 1], bi_feature[:, 0]], dim=1)

        features = self.convs(torch.cat([t1t2, t2t1], dim=0))
        logits = self.head(features)

        c12, c21 = torch.split(logits, batch_size, dim=0)

        return c12, c21


class ChangeViT(Module):
    """ChangeViT model for change detection.

    ChangeViT implementation using plain Vision Transformer as backbone
    with detail capture module and feature injection mechanism.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2406.12847

    .. note::
       For best results on LEVIR-CD as reported in the paper, use:

       * Backbone: ``vit_large_patch16_dinov3.sat493m`` (DINOv3-Large pretrained on
         satellite imagery)
       * Loss: Combined BCE+Dice loss (not yet implemented in ChangeDetectionTask)
       * Training: 80k steps with batch size 48
       * Image size: 256x256 patches

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        backbone: str,
        img_size: int = 256,
        in_channels: int = 3,
        num_classes: int = 1,
        pretrained: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize ChangeViT model.

        Args:
            backbone: Name of the timm ViT model to use as backbone
                (e.g., 'vit_small_patch14_dinov2', 'vit_tiny_patch16_224')
            img_size: Input image size (default: 256)
            in_channels: Number of input channels per temporal frame (default: 3)
            num_classes: Number of output classes (default: 1)
            pretrained: Whether to load pretrained weights from timm (default: False)
            **kwargs: Additional keyword arguments passed to timm backbone
        """
        super().__init__()

        self.encoder: Any = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            img_size=img_size,
            dynamic_img_size=True,
            in_chans=in_channels,
            **kwargs,
        )

        embed_dim: int = self.encoder.embed_dim  # type: ignore[assignment]

        self.detail_capture = DetailCaptureModule(
            in_channels=in_channels * 2, pretrained=pretrained
        )
        self.feature_injector = FeatureInjector(
            vit_dim=embed_dim, detail_dims=(64, 128, 256)
        )
        self.decoder = ChangeViTDecoder(in_channels=embed_dim, num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of ChangeViT.

        Args:
            x: Bitemporal input tensor [B, T, C, H, W]

        Returns:
            Change detection logits [B, 1, H, W]
        """
        _b, _t, _c, h, w = x.shape

        x_t1 = x[:, 0]
        x_t2 = x[:, 1]
        x_concat = rearrange(x, 'b t c h w -> b (t c) h w')

        vit_features_t1 = self.encoder.forward_features(x_t1)
        vit_features_t2 = self.encoder.forward_features(x_t2)

        detail_features = self.detail_capture(x_concat)

        patch_size_attr = self.encoder.patch_embed.patch_size
        patch_size = (
            patch_size_attr[0]
            if isinstance(patch_size_attr, tuple)
            else patch_size_attr
        )

        h_patch, w_patch = h // patch_size, w // patch_size
        num_patch_tokens = h_patch * w_patch

        patch_features_t1 = vit_features_t1[:, 1 : 1 + num_patch_tokens]
        patch_features_t2 = vit_features_t2[:, 1 : 1 + num_patch_tokens]

        vit_features_stacked = torch.stack(
            [patch_features_t1, patch_features_t2], dim=1
        )

        enhanced_features_list = []
        for t_idx in range(2):
            enhanced_feat = self.feature_injector(
                vit_features_stacked[:, t_idx], detail_features
            )
            enhanced_features_list.append(enhanced_feat)

        enhanced_features_tensor = torch.stack(enhanced_features_list, dim=1)

        enhanced_spatial = rearrange(
            enhanced_features_tensor, 'b t (h w) d -> b t d h w', h=h_patch, w=w_patch
        )

        c12, _c21 = self.decoder(enhanced_spatial)

        target_size = (x.shape[-2], x.shape[-1])
        change_logits: Tensor = F.interpolate(
            c12, size=target_size, mode='bilinear', align_corners=False
        )

        return change_logits
