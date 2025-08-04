# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""ChangeViT model implementation.

Based on the paper: https://arxiv.org/pdf/2406.12847
"""

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.modules import Module
from torchvision.models._api import Weights, WeightsEnum


class BasicBlock(Module):
    """ResNet BasicBlock with residual connection.

    Paper explicitly states: 'three residual convolutional blocks (C2-C4) adapted from ResNet18'
    """

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Module | None = None,
    ) -> None:
        """Initialize BasicBlock.

        Args:
            inplanes: Number of input channels
            planes: Number of output channels
            stride: Convolution stride
            downsample: Optional downsample module for residual connection
        """
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through BasicBlock with residual connection.

        Args:
            x: Input tensor

        Returns:
            Output tensor after residual block processing
        """
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        result: Tensor = self.relu(out)

        return result


class DetailCaptureModule(Module):
    """Detail capture module implementing ResNet18 C2-C4 layers.

    Paper states: 'three residual convolutional blocks (C2-C4) adapted from ResNet18'
    that generate 'three-scale detailed features: 1/2, 1/4, and 1/8 resolutions'
    with '2.7M parameters'.
    """

    def __init__(self, in_channels: int = 6) -> None:
        """Initialize the detail capture module.

        Args:
            in_channels: Number of input channels (typically 6 for bitemporal RGB).
        """
        super().__init__()
        self.inplanes = 64

        # Initial convolution like ResNet conv1 + bn1 + relu
        # Modified for 6-channel input (bitemporal RGB)
        self.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # ResNet18 layers adapted for detail capture
        # layer1: C2 - 1/2 scale (after conv1 stride=2)
        self.layer1 = self._make_layer(64, 2, stride=1)  # No additional downsampling

        # layer2: C3 - 1/4 scale
        self.layer2 = self._make_layer(128, 2, stride=2)  # Downsample 2x

        # layer3: C4 - 1/8 scale
        self.layer3 = self._make_layer(256, 2, stride=2)  # Downsample 2x

    def _make_layer(self, planes: int, blocks: int, stride: int = 1) -> nn.Sequential:
        """Create a ResNet layer with residual blocks."""
        downsample = None
        if stride != 1 or self.inplanes != planes * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes,
                    planes * BasicBlock.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass returning multi-scale features.

        Args:
            x: Input tensor of shape [B, C, H, W]

        Returns:
            Tuple of (C2, C3, C4) features at 1/2, 1/4, 1/8 scales
        """
        # Initial processing
        x = self.conv1(x)  # stride=2, so this gives us 1/2 scale
        x = self.bn1(x)
        x = self.relu(x)

        # ResNet layers for multi-scale features
        c2 = self.layer1(x)  # 1/2 scale (64 channels)
        c3 = self.layer2(c2)  # 1/4 scale (128 channels)
        c4 = self.layer3(c3)  # 1/8 scale (256 channels)

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

        # Cross-attention blocks for each scale
        self.cross_attns = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=vit_dim, num_heads=num_heads, batch_first=True
                )
                for _ in range(3)
            ]
        )

        # Projection layers to match ViT dimension
        self.detail_projs = nn.ModuleList(
            [nn.Linear(dim, vit_dim) for dim in detail_dims]
        )

        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(vit_dim * 4, vit_dim),  # ViT + 3 detail scales
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
        import torch.nn.functional as F

        b, n, d = vit_feats.shape
        enhanced_feats = [vit_feats]

        # Calculate target spatial dimensions from ViT patch count
        # Assume square patch grid: N = H_patch * W_patch
        patch_grid_size = int(n**0.5)  # sqrt(N) for square grid
        target_spatial = (patch_grid_size, patch_grid_size)

        for i, (detail_feat, cross_attn, proj) in enumerate(
            zip(detail_feats, self.cross_attns, self.detail_projs)
        ):
            # Spatially align detail features to match ViT patch resolution
            # This reduces attention complexity from O(N*H*W) to O(N*N)
            detail_aligned = F.adaptive_avg_pool2d(detail_feat, target_spatial)

            # Flatten spatial dimensions: [B, C, H_patch, W_patch] -> [B, N, C]
            detail_flat = detail_aligned.flatten(2).transpose(1, 2)

            # Project to ViT dimension
            detail_proj = proj(detail_flat)

            # Cross-attention: ViT as query, aligned detail as key/value
            enhanced_feat, _ = cross_attn(
                query=vit_feats, key=detail_proj, value=detail_proj
            )
            enhanced_feats.append(enhanced_feat)

        # Concatenate and fuse all features
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
        in_channels: int = 768,  # ViT embedding dimension
        inner_channels: int = 64,
        num_convs: int = 3,
        scale_factor: float = 16.0,  # ViT patch size
    ) -> None:
        """Initialize the ChangeViTDecoder.

        Args:
            in_channels: Input feature dimension (ViT embedding dim)
            inner_channels: Number of inner channels for processing
            num_convs: Number of convolutional layers
            scale_factor: Upsampling factor (should match ViT patch size)
        """
        super().__init__()

        # Feature processing layers
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

        # Classification layer and upsampling
        cls_layer = nn.Sequential(nn.Conv2d(inner_channels, 1, 3, 1, 1))
        upsample_layer = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        )

        layers.append(cls_layer)
        layers.append(upsample_layer)

        self.convs = nn.Sequential(*layers)

    def forward(self, bi_feature: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass for change detection.

        Args:
            bi_feature: Bitemporal features [B, T, C, H, W]

        Returns:
            List of bidirectional change predictions
        """
        batch_size = bi_feature.size(0)

        # Concatenate features in both temporal orders
        t1t2 = torch.cat([bi_feature[:, 0], bi_feature[:, 1]], dim=1)
        t2t1 = torch.cat([bi_feature[:, 1], bi_feature[:, 0]], dim=1)

        # Process both orderings together
        c1221 = self.convs(torch.cat([t1t2, t2t1], dim=0))
        c12, c21 = torch.split(c1221, batch_size, dim=0)

        return c12, c21


class ChangeViT(Module):
    """ChangeViT model for change detection.

    ChangeViT implementation using plain Vision Transformer as backbone
    with detail capture module and feature injection mechanism.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2406.12847
    """

    def __init__(
        self,
        vit_backbone: Any,
        detail_capture: DetailCaptureModule,
        feature_injector: FeatureInjector,
        decoder: ChangeViTDecoder,
        inference_mode: str = 't1t2',
    ) -> None:
        """Initialize ChangeViT model.

        Args:
            vit_backbone: Vision Transformer backbone (without classification head)
            detail_capture: Detail capture module for fine-grained features
            feature_injector: Feature injector for cross-attention fusion
            decoder: Change detection decoder for final prediction
            inference_mode: Inference mode ('t1t2', 't2t1', or 'mean')
        """
        super().__init__()

        self.vit_backbone = vit_backbone
        self.detail_capture = detail_capture
        self.feature_injector = feature_injector
        self.decoder = decoder

        if inference_mode not in ['t1t2', 't2t1', 'mean']:
            raise ValueError(f'Unknown inference_mode: {inference_mode}')
        self.inference_mode = inference_mode

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """Forward pass of ChangeViT.

        Args:
            x: Bitemporal input tensor [B, T, C, H, W]

        Returns:
            Dictionary containing change detection results
        """
        b, t, c, h, w = x.shape

        # Separate temporal frames for parallel processing
        x_t1 = x[:, 0]  # [B, C, H, W] - first temporal frame
        x_t2 = x[:, 1]  # [B, C, H, W] - second temporal frame
        x_concat = rearrange(
            x, 'b t c h w -> b (t c) h w'
        )  # [B, 2*C, H, W] for detail capture

        # Process each temporal frame separately through ViT backbone (parallel processing)
        vit_features_t1 = self.vit_backbone.forward_features(x_t1)  # [B, N, D]
        vit_features_t2 = self.vit_backbone.forward_features(x_t2)  # [B, N, D]

        if not isinstance(vit_features_t1, Tensor) or not isinstance(
            vit_features_t2, Tensor
        ):
            raise TypeError('ViT backbone forward_features must return a Tensor')

        # Extract detail features from concatenated input
        detail_features = self.detail_capture(x_concat)

        # Remove CLS tokens (first token) from each temporal frame
        patch_features_t1 = vit_features_t1[:, 1:]  # [B, N-1, D]
        patch_features_t2 = vit_features_t2[:, 1:]  # [B, N-1, D]

        # Stack temporal features for feature injection
        vit_features_stacked = torch.stack(
            [patch_features_t1, patch_features_t2], dim=1
        )  # [B, T=2, N-1, D]

        # Apply feature injection to each temporal frame
        enhanced_features_list = []
        for t_idx in range(2):
            enhanced_feat = self.feature_injector(
                vit_features_stacked[:, t_idx], detail_features
            )
            enhanced_features_list.append(enhanced_feat)

        enhanced_features_tensor = torch.stack(
            enhanced_features_list, dim=1
        )  # [B, T=2, N-1, D]

        # Get patch size from backbone
        patch_embed = getattr(self.vit_backbone, 'patch_embed', None)
        if patch_embed is None:
            raise AttributeError('ViT backbone must have patch_embed attribute')

        patch_size_attr = getattr(patch_embed, 'patch_size', None)
        if patch_size_attr is None:
            raise AttributeError('patch_embed must have patch_size attribute')

        if isinstance(patch_size_attr, list | tuple):
            patch_size = patch_size_attr[0]
        else:
            patch_size = patch_size_attr

        h_patch, w_patch = h // patch_size, w // patch_size

        # Reshape to spatial format for each temporal frame
        enhanced_spatial = rearrange(
            enhanced_features_tensor, 'b t (h w) d -> b t d h w', h=h_patch, w=w_patch
        )

        # Apply change detection with proper temporal features
        c12, c21 = self.decoder(
            enhanced_spatial
        )  # Returns tuple of [B, 1, H, W] tensors

        change_logits = c12  # Match target format [B, 1, H, W]

        # Handle training vs inference mode
        if self.training:
            # Training: return logits for BCE loss computation
            return {
                'change_prob': change_logits
            }  # Actually logits, but keeping key name for compatibility
        else:
            # Inference: return probabilities and binary map
            change_prob = torch.sigmoid(change_logits)
            change_binary = (change_prob > 0.5).float()  # Threshold at 0.5
            return {'change_prob': change_prob, 'change_binary': change_binary}


class ChangeViT_Weights(WeightsEnum):  # type: ignore[misc]
    """ChangeViT model weights.

    .. versionadded:: 0.9
    """

    # DeiT pre-trained weights (as used in paper)
    DEIT_TINY = Weights(
        url=None,
        transforms=None,
        meta={
            'model': 'changevit_tiny',
            'backbone': 'deit_tiny_patch16_224',
            'pretrained': True,
            'paper': 'ChangeViT: Unleashing Plain Vision Transformers for Change Detection',
        },
    )

    DEIT_SMALL = Weights(
        url=None,
        transforms=None,
        meta={
            'model': 'changevit_small',
            'backbone': 'deit_small_patch16_224',
            'pretrained': True,
            'paper': 'ChangeViT: Unleashing Plain Vision Transformers for Change Detection',
        },
    )

    # DINOv2 pre-trained weights (alternative initialization)
    DINOV2_SMALL = Weights(
        url=None,
        transforms=None,
        meta={
            'model': 'changevit_small',
            'backbone': 'vit_small_patch14_dinov2',
            'pretrained': True,
            'paper': 'ChangeViT: Unleashing Plain Vision Transformers for Change Detection',
        },
    )


def changevit_small(
    weights: ChangeViT_Weights | None = None, **kwargs: Any
) -> ChangeViT:
    """ChangeViT Small model.

    Uses ViT-Small as backbone with detail capture module.

    Args:
        weights: Pre-trained model weights to use.
        **kwargs: Additional keyword arguments.

    Returns:
        A ChangeViT model.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            '`timm` is not installed and is required for this model. '
            'Please install it with `pip install timm`.'
        ) from e

    # Create ViT backbone from timm - use DINOv2 small for best performance per paper
    vit_backbone = timm.create_model(
        'vit_small_patch14_dinov2',
        pretrained=weights is not None,
        num_classes=0,  # Remove classification head
        img_size=224,  # Override default 518x518 to work with 224x224 inputs
        **kwargs,
    )

    # Use standard 3-channel patch embedding

    # Get embed_dim safely
    embed_dim = getattr(vit_backbone, 'embed_dim', None)
    if embed_dim is None:
        raise AttributeError('ViT backbone must have embed_dim attribute')

    # Get actual patch size for proper upsampling
    patch_embed = getattr(vit_backbone, 'patch_embed', None)
    if patch_embed is not None and hasattr(patch_embed, 'patch_size'):
        patch_size_attr = getattr(patch_embed, 'patch_size')
        if isinstance(patch_size_attr, list | tuple):
            patch_size = patch_size_attr[0]
        else:
            patch_size = patch_size_attr
    else:
        patch_size = 14  # Default for DINOv2

    # Create components
    detail_capture = DetailCaptureModule(in_channels=6)
    feature_injector = FeatureInjector(vit_dim=embed_dim, detail_dims=(64, 128, 256))
    decoder = ChangeViTDecoder(
        in_channels=embed_dim,
        scale_factor=float(patch_size),  # Use actual patch size (14 for DINOv2)
    )

    # Create model
    model = ChangeViT(
        vit_backbone=vit_backbone,
        detail_capture=detail_capture,
        feature_injector=feature_injector,
        decoder=decoder,
    )

    # Load weights if provided
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True))

    return model


def changevit_tiny(
    weights: ChangeViT_Weights | None = None, **kwargs: Any
) -> ChangeViT:
    """ChangeViT Tiny model.

    Uses ViT-Tiny as backbone with detail capture module.
    Paper implementation: ChangeViT-T

    Args:
        weights: Pre-trained model weights to use.
        **kwargs: Additional keyword arguments.

    Returns:
        A ChangeViT model.
    """
    try:
        import timm
    except ImportError as e:
        raise ImportError(
            '`timm` is not installed and is required for this model. '
            'Please install it with `pip install timm`.'
        ) from e

    # Create ViT backbone from timm - use DeiT tiny for paper reproduction
    vit_backbone = timm.create_model(
        'deit_tiny_patch16_224',
        pretrained=weights is not None,
        num_classes=0,  # Remove classification head
        **kwargs,
    )

    # Use standard 3-channel patch embedding

    # Get embed_dim safely
    embed_dim = getattr(vit_backbone, 'embed_dim', None)
    if embed_dim is None:
        raise AttributeError('ViT backbone must have embed_dim attribute')

    # Get actual patch size for proper upsampling
    patch_embed = getattr(vit_backbone, 'patch_embed', None)
    if patch_embed is not None and hasattr(patch_embed, 'patch_size'):
        patch_size_attr = getattr(patch_embed, 'patch_size')
        if isinstance(patch_size_attr, list | tuple):
            patch_size = patch_size_attr[0]
        else:
            patch_size = patch_size_attr
    else:
        patch_size = 16  # Default for DeiT

    # Create components
    detail_capture = DetailCaptureModule(in_channels=6)
    feature_injector = FeatureInjector(vit_dim=embed_dim, detail_dims=(64, 128, 256))
    decoder = ChangeViTDecoder(
        in_channels=embed_dim,
        scale_factor=float(patch_size),  # Use actual patch size (16 for DeiT)
    )

    # Create model
    model = ChangeViT(
        vit_backbone=vit_backbone,
        detail_capture=detail_capture,
        feature_injector=feature_injector,
        decoder=decoder,
    )

    # Load weights if provided
    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=True))

    return model
