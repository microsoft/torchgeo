# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Swin v2 Transformer models."""

from typing import Any, Optional

import kornia.augmentation as K
import torch
import torchvision
from torchvision.models import SwinTransformer
from torchvision.models._api import Weights, WeightsEnum

from ..transforms import AugmentationSequential

__all__ = ["Swin_V2_B_Weights"]


# https://github.com/allenai/satlas/blob/bcaa968da5395f675d067613e02613a344e81415/satlas/cmd/model/train.py#L42 # noqa: E501
# All Satlas imagery is uint8 and normalized to the range (0, 1) by dividing by 255
_satlas_transforms = AugmentationSequential(
    K.CenterCrop(256),
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
    data_keys=["image"],
)

# https://github.com/pytorch/vision/pull/6883
# https://github.com/pytorch/vision/pull/7107
# Can be removed once torchvision>=0.15 is required
Weights.__deepcopy__ = lambda *args, **kwargs: args[0]


class Swin_V2_B_Weights(WeightsEnum):  # type: ignore[misc]
    """Swin Transformer v2 Base weights.

    For `torchvision <https://github.com/pytorch/vision>`_
    *swin_v2_b* implementation.

    .. versionadded:: 0.6
    """

    NAIP_RGB_SATLAS = Weights(
        url="https://huggingface.co/torchgeo/swin_v2_b_naip_rgb_satlas/resolve/main/swin_v2_b_naip_rgb_satlas-685f45bd.pth",  # noqa: E501
        transforms=_satlas_transforms,
        meta={
            "dataset": "Satlas",
            "in_chans": 3,
            "model": "swin_v2_b",
            "publication": "https://arxiv.org/abs/2211.15660",
            "repo": "https://github.com/allenai/satlas",
        },
    )

    SENTINEL2_RGB_SATLAS = Weights(
        url="https://huggingface.co/torchgeo/swin_v2_b_sentinel2_rgb_satlas/resolve/main/swin_v2_b_sentinel2_rgb_satlas-51471041.pth",  # noqa: E501
        transforms=_satlas_transforms,
        meta={
            "dataset": "Satlas",
            "in_chans": 3,
            "model": "swin_v2_b",
            "publication": "https://arxiv.org/abs/2211.15660",
            "repo": "https://github.com/allenai/satlas",
        },
    )


def swin_v2_b(
    weights: Optional[Swin_V2_B_Weights] = None, *args: Any, **kwargs: Any
) -> SwinTransformer:
    """Swin Transformer v2 base model.

    If you use this model in your research, please cite the following paper:

    * https://arxiv.org/abs/2111.09883

    .. versionadded:: 0.6

    Args:
        weights: Pre-trained model weights to use.
        *args: Additional arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.
        **kwargs: Additional keywork arguments to
            pass to :class:`torchvision.models.swin_transformer.SwinTransformer`.

    Returns:
        A Swin Transformer Base model.
    """
    model: SwinTransformer = torchvision.models.swin_v2_b(weights=None, *args, **kwargs)

    if weights:
        model.load_state_dict(weights.get_state_dict(progress=True), strict=False)

    return model
