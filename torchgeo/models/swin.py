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


# Custom transform for Sentinel-2 multispectral model inputs; Uses 9 bands.
class MSS2Transform(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Divide the first 3 channels by 255
        x[:, :3, :, :] = x[:, :3, :, :] / 255.0
        # Divide the last 6 channels by 8160 and clip to 0-1
        x[:, -6:, :, :] = torch.clamp(x[:, -6:, :, :] / 8160.0, 0.0, 1.0)
        return x


# https://github.com/allenai/satlas/blob/bcaa968da5395f675d067613e02613a344e81415/satlas/cmd/model/train.py#L42 # noqa: E501
# Satlas Sentinel-1 and RGB Sentinel-2 and NAIP imagery is uint8 and is normalized to (0, 1) by dividing by 255. # noqa: E501
_satlas_transforms = AugmentationSequential(
    K.CenterCrop(256),
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)),
    data_keys=["image"],
)

# Satlas multispectral Sentinel-2 imagery divides first 3 bands by 255 and the following 6 bands by 8160, both clipped to (0, 1). # noqa: E501
_sentinel2_ms_satlas_transforms = AugmentationSequential(
    K.CenterCrop(256), MSS2Transform(), data_keys=["image"]
)

# Satlas Landsat imagery is 16-bit, normalized by clipping some pixel N with (N-4000)/16320 to (0, 1). # noqa: E501
_landsat_satlas_transforms = AugmentationSequential(
    K.CenterCrop(256),
    K.Normalize(mean=torch.tensor(4000), std=torch.tensor(1)),
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(16320)),
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

    NAIP_RGB_SI_SATLAS = Weights(
        url="https://huggingface.co/allenai/satlas-pretrain/resolve/main/aerial_swinb_si.pth",  # noqa: E501
        transforms=_satlas_transforms,
        meta={
            "dataset": "Satlas",
            "in_chans": 3,
            "model": "swin_v2_b",
            "publication": "https://arxiv.org/abs/2211.15660",
            "repo": "https://github.com/allenai/satlas",
        },
    )

    SENTINEL2_RGB_SI_SATLAS = Weights(
        url="https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_rgb.pth",  # noqa: E501
        transforms=_satlas_transforms,
        meta={
            "dataset": "Satlas",
            "in_chans": 3,
            "model": "swin_v2_b",
            "publication": "https://arxiv.org/abs/2211.15660",
            "repo": "https://github.com/allenai/satlas",
        },
    )

    SENTINEL2_MS_SI_SATLAS = Weights(
        url="https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel2_swinb_si_ms.pth",  # noqa: E501
        transforms=_sentinel2_ms_satlas_transforms,
        meta={
            "dataset": "Satlas",
            "in_chans": 9,
            "model": "swin_v2_b",
            "publication": "https://arxiv.org/abs/2211.15660",
            "repo": "https://github.com/allenai/satlas",
        },
    )

    SENTINEL1_SI_SATLAS = Weights(
        url="https://huggingface.co/allenai/satlas-pretrain/resolve/main/sentinel1_swinb_si.pth",  # noqa: E501
        transforms=_satlas_transforms,
        meta={
            "dataset": "Satlas",
            "in_chans": 2,
            "model": "swin_v2_b",
            "publication": "https://arxiv.org/abs/2211.15660",
            "repo": "https://github.com/allenai/satlas",
        },
    )

    LANDSAT_SI_SATLAS = Weights(
        url="https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_si.pth",  # noqa: E501
        transforms=_landsat_satlas_transforms,
        meta={
            "dataset": "Satlas",
            "in_chans": 2,
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
