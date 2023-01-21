# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Vision Transformer models."""

from typing import Any, Optional

import kornia.augmentation as K
import timm
from timm.models.vision_transformer import VisionTransformer
from torchvision.models._api import Weights, WeightsEnum

from ..transforms import AugmentationSequential

__all__ = ["ViTSmall16_Weights"]

_zhu_xlab_transforms = AugmentationSequential(
    K.Resize(256), K.CenterCrop(224), data_keys=["image"]
)

# https://github.com/pytorch/vision/pull/6883
# https://github.com/pytorch/vision/pull/7107
# Can be removed once torchvision>=0.15 is required
Weights.__deepcopy__ = lambda *args, **kwargs: args[0]


class ViTSmall16_Weights(WeightsEnum):  # type: ignore[misc]
    """Vision Transformer Samll Patch Size 16 weights.

    For `timm <https://github.com/rwightman/pytorch-image-models>`_
    *vit_small_patch16_224* implementation.

    .. versionadded:: 0.4
    """

    SENTINEL2_ALL_MOCO = Weights(
        url=(
            "https://huggingface.co/torchgeo/vit_small_patch16_224_sentinel2_all_moco/"
            "resolve/main/vit_small_patch16_224_sentinel2_all_moco.pth"
        ),
        transforms=_zhu_xlab_transforms,
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
        transforms=_zhu_xlab_transforms,
        meta={
            "dataset": "SSL4EO-S12",
            "in_chans": 13,
            "model": "vit_small_patch16_224",
            "publication": "https://arxiv.org/abs/2211.07044",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "ssl_method": "dino",
        },
    )


def vit_small_patch16_224(
    weights: Optional[ViTSmall16_Weights] = None, *args: Any, **kwargs: Any
) -> VisionTransformer:
    """Vision Transform (ViT) small patch size 16 model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2010.11929

    .. versionadded:: 0.4

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to pass to :func:`timm.create_model`.
        **kwargs: Additional keywork arguments to pass to :func:`timm.create_model`.

    Returns:
        A ViT small 16 model.
    """
    if weights:
        kwargs["in_chans"] = weights.meta["in_chans"]

    model: VisionTransformer = timm.create_model(
        "vit_small_patch16_224", *args, **kwargs
    )

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    return model
