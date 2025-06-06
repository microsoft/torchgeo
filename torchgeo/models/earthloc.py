# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""EarthLoc."""

import math
from typing import Any

import kornia.augmentation as K
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

# Note the images used are Sentinel-2 Cloudless RGB Mosaics from https://s2maps.eu/
_earthloc_sentinel2_bands = ['B4', 'B3', 'B2']

# https://github.com/gmberton/EarthLoc/blob/2da231ae7ec9764fac6cde2aa88a17db23c1bb6a/datasets/train_dataset.py#L43
# https://github.com/gmberton/EarthLoc/blob/2da231ae7ec9764fac6cde2aa88a17db23c1bb6a/augmentations.py#L40
# Divide by 255 and normalize with ImageNet mean and std
_earthloc_transforms = K.AugmentationSequential(
    K.Normalize(mean=torch.tensor(0.0), std=torch.tensor(255.0)),
    K.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225]),
    ),
    K.Resize((320, 320)),
    data_keys=None,
)


class FeatureMixerLayer(nn.Module):
    """Feature Mixer Layer in the MixVPR architecture.

    Adapted from https://github.com/gmberton/EarthLoc. Copyright (c) 2024 Gabriele Berton

    .. versionadded:: 0.8
    """

    def __init__(self, input_dim: int, mlp_ratio: int = 1) -> None:
        """Initialize the FeatureMixerLayer.

        Args:
            input_dim: Input dimension of the feature maps.
            mlp_ratio: Ratio of the mid projection layer in the mlp mixer block.
        """
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, int(input_dim * mlp_ratio)),
            nn.ReLU(),
            nn.Linear(int(input_dim * mlp_ratio), input_dim),
        )

        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the FeatureMixerLayer.

        Args:
            x: Input tensor of shape (batch_size, num_features, feature_dim).

        Returns:
            Output tensor after applying the feature mixer.
        """
        x = x + self.mix(x)
        return x


class MixVPR(nn.Module):
    """MixVPR model for generating feature descriptors.

    Adapted from https://github.com/gmberton/EarthLoc. Copyright (c) 2024 Gabriele Berton

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2303.02190

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        in_channels: int = 1024,
        in_h: int = 20,
        in_w: int = 20,
        out_channels: int = 512,
        mix_depth: int = 1,
        mlp_ratio: int = 1,
        out_rows: int = 4,
    ) -> None:
        """Initialize the MixVPR model.

        Args:
            in_channels: Number of input channels in the feature maps.
            in_h: Height of the input feature maps.
            in_w: Width of the input feature maps.
            out_channels: Number of output channels after depth-wise projection.
            mix_depth: Number of stacked FeatureMixer layers.
            mlp_ratio: Ratio of the mid projection layer in the mixer block.
            out_rows: Row-wise projection dimension.
        """
        super().__init__()
        self.in_h = in_h
        self.in_w = in_w
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_rows = out_rows
        self.mix_depth = mix_depth
        self.mlp_ratio = mlp_ratio

        hw = in_h * in_w
        self.mix = nn.Sequential(
            *[
                FeatureMixerLayer(input_dim=hw, mlp_ratio=mlp_ratio)
                for _ in range(self.mix_depth)
            ]
        )
        self.channel_proj = nn.Linear(in_channels, out_channels)
        self.row_proj = nn.Linear(hw, out_rows)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the MixVPR encoder.

        Args:
            x: Input 2D image embeddings of shape (b, c, h, w).

        Returns:
            Output feature descriptor tensor of shape (b, d).
        """
        x = rearrange(x, 'b c h w -> b c (h w)')
        x = self.mix(x)
        x = rearrange(x, 'b c d -> b d c')
        x = self.channel_proj(x)
        x = rearrange(x, 'b d c -> b c d')
        x = self.row_proj(x)
        x = rearrange(x, 'b c d -> b (c d)')
        x = F.normalize(x, p=2, dim=1)
        return x


class EarthLoc(nn.Module):
    """EarthLoc model for generating feature descriptors from satellite imagery.

    Adapted from https://github.com/gmberton/EarthLoc. Copyright (c) 2024 Gabriele Berton

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.06758

    .. versionadded:: 0.8
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 320,
        desc_dim: int = 4096,
        backbone: str = 'resnet50',
        pretrained: bool = True,
    ) -> None:
        """Initialize the EarthLoc model.

        Args:
            in_channels: Number of input channels in the images (default: 3 for RGB).
            image_size: Size of the input images (assumed square).
            desc_dim: Dimension of the final output feature descriptor.
            backbone: Backbone model to use for feature extraction (default: "resnet50").
            pretrained: Whether to use pre-trained weights for the backbone model.
        """
        super().__init__()
        self.image_size = image_size
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_channels,
            num_classes=0,
            global_pool='',
        )
        self.backbone.layer4 = nn.Identity()
        out_channels = desc_dim // 4
        self.aggregator = MixVPR(
            in_channels=1024,
            in_h=math.ceil(image_size / 16),
            in_w=math.ceil(image_size / 16),
            out_channels=out_channels,
            mix_depth=4,
            mlp_ratio=1,
            out_rows=4,
        )
        self.fc = nn.Linear(desc_dim, desc_dim)
        self.desc_dim = desc_dim  # Dimension of final descriptor

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the EarthLoc model.

        Args:
            x: Input tensor of shape (b, c, h, w).

        Returns:
            Output feature descriptor tensor of shape (b, desc_dim).
        """
        x = self.backbone(x)
        x = self.aggregator(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class EarthLoc_Weights(WeightsEnum):  # type: ignore[misc]
    """EarthLoc weights."""

    SENTINEL2_RESNET50 = Weights(
        url='https://huggingface.co/torchgeo/earthloc/resolve/53a4bb90a7754b12f44986521ac7a711b4795959/earthloc-8b632e30.pth',
        transforms=_earthloc_transforms,
        meta={
            'dataset': 'EarthLoc',
            'in_chans': 3,
            'image_size': 320,
            'desc_dim': 4096,
            'encoder': 'resnet50',
            'bands': _earthloc_sentinel2_bands,
            'publication': 'https://arxiv.org/abs/2403.06758',
            'repo': 'https://github.com/gmberton/EarthLoc',
        },
    )


def earthloc(
    weights: EarthLoc_Weights | None = None, *args: Any, **kwargs: Any
) -> EarthLoc:
    """EarthLoc model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2403.06758

    .. versionadded:: 0.8

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :class:`EarthLoc`.
        **kwargs: Additional keyword arguments to pass to :class:`EarthLoc`.

    Returns:
        An EarthLoc model.
    """
    if weights:
        kwargs |= {
            'in_channels': weights.meta['in_chans'],
            'image_size': weights.meta['image_size'],
            'desc_dim': weights.meta['desc_dim'],
            'backbone': weights.meta['encoder'],
            'pretrained': False,
        }
        model = EarthLoc(*args, **kwargs)
        model.load_state_dict(weights.get_state_dict(progress=True), strict=True)
    else:
        model = EarthLoc(*args, **kwargs)

    return model
