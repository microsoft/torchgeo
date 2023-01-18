# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained ResNet models."""

import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

__all__ = ["ResNet50_Weights", "ResNet18_Weights"]


class ResNet18_Weights(WeightsEnum):
    """ResNet18 weights.

    For `timm
    <https://github.com/rwightman/pytorch-image-models>`_
    *resnet18* implementation.
    """

    SENTINEL2_ALL_MOCO = Weights(
        url=(
            "https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco/"
            "resolve/main/resnet18_sentinel2_all_moco.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 13,
            "model": "resnet18",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "moco",
        },
    )

    SENTINEL2_RGB_MOCO = Weights(
        url=(
            "https://huggingface.co/torchgeo/resnet18_sentinel2_rgb_moco/"
            "resolve/main/resnet18_sentinel2_rgb_moco.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 3,
            "model": "resnet18",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "moco",
        },
    )


class ResNet50_Weights(WeightsEnum):
    """ResNet50 weights.

    For `timm
    <https://github.com/rwightman/pytorch-image-models>`_
    *resnet50* implementation.
    """

    SENTINEL1_ALL_MOCO = Weights(
        url=(
            "https://huggingface.co/torchgeo/resnet50_sentinel1_all_moco/"
            "resolve/main/resnet50_sentinel1_all_moco.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 2,
            "model": "resnet50",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "moco",
        },
    )

    SENTINEL2_ALL_MOCO = Weights(
        url=(
            "https://huggingface.co/torchgeo/resnet50_sentinel2_all_moco/"
            "resolve/main/resnet50_sentinel2_all_moco.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 13,
            "model": "resnet50",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "moco",
        },
    )

    SENTINEL2_RGB_MOCO = Weights(
        url=(
            "https://huggingface.co/torchgeo/resnet50_sentinel2_rgb_moco/"
            "resolve/main/resnet50_sentinel2_rgb_moco.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 3,
            "model": "resnet50",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "moco",
        },
    )

    SENTINEL2_ALL_DINO = Weights(
        url=(
            "https://huggingface.co/torchgeo/resnet50_sentinel2_all_dino/"
            "resolve/main/resnet50_sentinel2_all_dino.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 13,
            "model": "resnet50",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "dino",
        },
    )
