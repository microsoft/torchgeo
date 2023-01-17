# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained Vision Transformer models."""

import os
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

from torchgeo.models.utils import adjust_dino_weights_zhu_lab, load_state_dict_from_url

__all__ = ["VITSmall16_Weights"]


class VITSmall16_Weights(WeightsEnum):
    """Vision Transformer Samll Patch Size 16 weights.

    For `timm
    <https://github.com/rwightman/pytorch-image-models>`_
    *vit_small_patch16_224* implementation.
    """

    SENTINEL2_ALL_MOCO = Weights(
        url=(
            "https://drive.google.com/file/d/1Tx07L6OilkfcgE2HWiSXHRmRepCPdn6V/"
            "view?usp=sharing/B13_vits16_moco_0099_ckpt.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "ssl_method": "moco",
            "publication": "https://arxiv.org/abs/2211.07044",
            "dataset": "SSL4EO-S12",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "in_chans": 13,
        },
    )

    SENTINEL2_ALL_DINO = Weights(
        url=(
            "https://drive.google.com/file/d/1CseO5vvMReGlAulm5o4ZgbjUgj8VlAH7/"
            "view?usp=sharing/B13_vits16_dino_0099_ckpt-pth"
        ),
        transforms=nn.Identity(),
        meta={
            "ssl_method": "dino",
            "publication": "https://arxiv.org/abs/2211.07044",
            "dataset": "SSL4EO-S12",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "in_chans": 13,
        },
    )

    def get_state_dict(self) -> Dict[str, Any]:
        """Retrieve pretrained weights state_dict."""
        root = os.path.join(torch.hub.get_dir(), "checkpoints")
        map_location = torch.device("cpu")

        if "SENTINEL2_ALL_MOCO" in str(self):
            filename = "vit_small_patch16_224_" + str(self).lower() + ".pth"
            ckpt = load_state_dict_from_url(root, filename, self.url, map_location)
            state_dict = adjust_moco_weights_zhu_lab(ckpt["state_dict"])

        elif "SENTINEL2_ALL_DINO" in str(self):
            filename = "vit_small_patch16_224_" + str(self).lower() + ".pth"
            ckpt = load_state_dict_from_url(root, filename, self.url, map_location)
            state_dict = adjust_dino_weights_zhu_lab(ckpt["teacher"])

        return state_dict


def adjust_moco_weights_zhu_lab(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Loading Moco VIT weights from https://github.com/zhu-xlab/SSL4EO-S12.

    # https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/
    # src/benchmark/transfer_classification/linear_BE_moco_v3.py#L199-L220
    """
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith("module.base_encoder") and not k.startswith(
            "module.base_encoder.%s" % "head"
        ):
            # remove prefix
            state_dict[k[len("module.base_encoder.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    return state_dict
