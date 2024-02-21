# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Swin v2 Transformer models."""

from typing import Any, Optional

import kornia.augmentation as K
import torch
import torchvision
from kornia.contrib import Lambda
from torchvision.models import SwinTransformer
from torchvision.models._api import Weights, WeightsEnum

from ..transforms import AugmentationSequential

__all__ = ["Swin_V2_B_Weights"]

# https://github.com/allenai/satlas/blob/bcaa968da5395f675d067613e02613a344e81415/satlas/cmd/model/train.py#L42 # noqa: E501
# Satlas uses the TCI product for Sentinel-2 RGB, which is in the range (0, 255). 
# See details:  https://github.com/allenai/satlas/blob/main/Normalization.md#sentinel-2-images.  # noqa: E501
# Satlas Sentinel-1 and RGB Sentinel-2 and NAIP imagery is uint8 and is normalized to (0, 1) by dividing by 255. # noqa: E501
_satlas_transforms = AugmentationSequential(
    K.Normalize(mean=torch.tensor(0), std=torch.tensor(255)), data_keys=["image"]
)

# Satlas uses the TCI product for Sentinel-2 RGB, which is in the range (0, 255).
# See details:  https://github.com/allenai/satlas/blob/main/Normalization.md#sentinel-2-images.  # noqa: E501
# Satlas Sentinel-2 multispectral imagery has first 3 bands divided by 255 and the following 6 bands by 8160, both clipped to (0, 1). # noqa: E501
_std = torch.tensor(
    [255.0, 255.0, 255.0, 8160.0, 8160.0, 8160.0, 8160.0, 8160.0, 8160.0]
)  # noqa: E501
_mean = torch.zeros_like(_std)
_sentinel2_ms_satlas_transforms = AugmentationSequential(
    K.Normalize(mean=_mean, std=_std),
    Lambda(lambda x: torch.clamp(x, min=0.0, max=1.0)),
    data_keys=["image"],
)

# Satlas Landsat imagery is 16-bit, normalized by clipping some pixel N with (N-4000)/16320 to (0, 1). # noqa: E501
_landsat_satlas_transforms = AugmentationSequential(
    K.Normalize(mean=torch.tensor(4000), std=torch.tensor(16320)),
    Lambda(lambda x: torch.clamp(x, min=0.0, max=1.0)),
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
            "bands": ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12"],
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
            "bands": ["VH", "VV"],
        },
    )

    LANDSAT_SI_SATLAS = Weights(
        url="https://huggingface.co/allenai/satlas-pretrain/resolve/main/landsat_swinb_si.pth",  # noqa: E501
        transforms=_landsat_satlas_transforms,
        meta={
            "dataset": "Satlas",
            "in_chans": 11,
            "model": "swin_v2_b",
            "publication": "https://arxiv.org/abs/2211.15660",
            "repo": "https://github.com/allenai/satlas",
            "bands": [
                "B01",
                "B02",
                "B03",
                "B04",
                "B05",
                "B06",
                "B07",
                "B08",
                "B09",
                "B10",
                "B11",
            ],  # noqa: E501
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
