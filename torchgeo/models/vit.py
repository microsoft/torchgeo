# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Vision Transformer models."""

import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

__all__ = ["ViTSmall16_Weights"]


class ViTSmall16_Weights(WeightsEnum):
    """Vision Transformer Samll Patch Size 16 weights.

    For `timm
    <https://github.com/rwightman/pytorch-image-models>`_
    *vit_small_patch16_224* implementation.
    """

    SENTINEL2_ALL_MOCO = Weights(
        url=(
            "https://huggingface.co/torchgeo/vit_small_patch16_224_sentinel2_all_moco/"
            "resolve/main/vit_small_patch16_224_sentinel2_all_moco.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 13,
            "model": "vit_small_patch16_224",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "moco",
        },
    )

    SENTINEL2_ALL_DINO = Weights(
        url=(
            "https://huggingface.co/torchgeo/vit_small_patch16_224_sentinel2_all_dino/"
            "resolve/main/vit_small_patch16_224_sentinel2_all_dino.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 13,
            "model": "vit_small_patch16_224",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "dino",
        },
    )
