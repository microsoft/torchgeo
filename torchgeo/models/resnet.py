# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Pre-trained ResNet models."""

import os
from typing import Any, Dict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models._api import Weights, WeightsEnum

from torchgeo.datasets.utils import download_url

__all__ = ["ResNet50_Weights", "ResNet18_Weights"]


class ResNet18_Weights(WeightsEnum):
    """ResNet18 weights.

    For `timm
    <https://github.com/rwightman/pytorch-image-models>`_ implementation.
    """

    SENTINEL2_RGB_MOCO = Weights(
        url=(
            "https://drive.google.com/file/d/1U_m39Owahk15Vg1uL1MYbPAmAyUWBKfI/"
            "view?usp=sharing/B3_rn18_moco_0199_ckpt.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "ssl_method": "moco",
            "publication": "https://arxiv.org/abs/2211.07044",
            "dataset": "SSL4EO-S12",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "num_input_channels": 3,
        },
    )

    SENTINEL2_ALL_MOCO = Weights(
        url=(
            "https://drive.google.com/file/d/1iWLm7ljQ6tKZiVp47pJUPDe3Un0BUd9o/"
            "view?usp=sharing/B13_rn18_moco_0099_ckpt.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "ssl_method": "moco",
            "publication": "https://arxiv.org/abs/2211.07044",
            "dataset": "SSL4EO-S12",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "num_input_channels": 13,
        },
    )

    def get_state_dict(self) -> Dict[str, Any]:
        """Retrieve pretrained weights state_dict."""
        root = os.path.join(torch.hub.get_dir(), "checkpoints")
        map_location = torch.device("cpu")

        if "SENTINEL2_ALL_MOCO" in str(self):
            filename = "resnet18_moco_sentinel2_all.pth"
            ckpt = load_state_dict_from_url(root, filename, self.url, map_location)
            state_dict = adjust_moco_weights_zhu_lab(ckpt["state_dict"])

        elif "SENTINEL2_RGB_MOCO" in str(self):
            filename = "resnet18_moco_sentinel2_rgb.pth"
            ckpt = load_state_dict_from_url(root, filename, self.url, map_location)
            state_dict = adjust_moco_weights_zhu_lab(ckpt["state_dict"])

        return state_dict


class ResNet50_Weights(WeightsEnum):
    """ResNet50 weights.

    For `timm
    <https://github.com/rwightman/pytorch-image-models>`_ implementation.
    """

    IMAGENET_RGB = Weights(
        url=(
            "https://github.com/rwightman/pytorch-image-models/releases/"
            "download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "dataset": "imagenet",
            "ssl_method": None,
            "repo": (
                "https://github.com/rwightman/pytorch-image-models/"
                "blob/main/timm/models/resnet.py"
            ),
            "num_input_channels": 3,
        },
    )
    SENTINEL2_RGB_SECO = Weights(
        url="https://zenodo.org/record/4728033/files/seco_resnet50_1m.ckpt?download=1",
        transforms=nn.Identity(),
        meta={
            "ssl_method": "seco",
            "dataset": "seco",
            "publication": "https://arxiv.org/abs/2103.16607",
            "repo": "https://github.com/ServiceNow/seasonal-contrast",
            "num_input_channels": 3,
        },
    )

    GOOGLEEARTH_MILLIONAID_RGB = Weights(
        url="https://drive.google.com/file/d/1K3P4_fDfcBRGqpKoSdSa6OXS4xC1xLC9",
        transforms=nn.Identity(),
        meta={
            "dataset": "millionaid",
            "ssl_method": None,
            "repo": (
                "https://github.com/ViTAE-Transformer/"
                "ViTAE-Transformer-Remote-Sensing"
            ),
            "num_input_channels": 3,
        },
    )

    SENTINEL2_ALL_MOCO = Weights(
        url=(
            "https://drive.google.com/file/d/1OrtPfG2wkO05bimstQ_T9Dza8z3zp8i-/"
            "view?usp=sharing/B13_rn50_moco_0099_ckpt.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "ssl_method": "moco",
            "publication": "https://arxiv.org/abs/2211.07044",
            "dataset": "SSL4EO-S12",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "num_input_channels": 13,
        },
    )

    SENTINEL2_RGB_MOCO = Weights(
        url=(
            "https://drive.google.com/file/d/1UEpA9sOcA47W0cmwQhkSeXfQxrL-EcJB/"
            "view?usp=sharing/B3_rn50_moco_0099_ckpt.path"
        ),
        transforms=nn.Identity(),
        meta={
            "ssl_method": "moco",
            "publication": "https://arxiv.org/abs/2211.07044",
            "dataset": "SSL4EO-S12",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "num_input_channels": 3,
        },
    )

    SENTINEL1_GRD_MOCO = Weights(
        url=(
            "https://drive.google.com/file/d/1gjTTWikf1qORJyFifWD1ksk9HzezqQ0b/"
            "view?usp=sharing/B2_moco_rn50_0099_ckpt.pth"
        ),
        transforms=nn.Identity(),
        meta={
            "ssl_method": "moco",
            "publication": "https://arxiv.org/abs/2211.07044",
            "dataset": "SSL4EO-S12",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "num_input_channels": 2,
        },
    )

    SENTINEL2_ALL_DINO = Weights(
        url="https://drive.google.com/file/d/1iSHHp_cudPjZlshqWXVZj5TK74P32a2q",
        transforms=nn.Identity(),
        meta={
            "ssl_method": "dino",
            "publication": "https://arxiv.org/abs/2211.07044",
            "dataset": "SSL4EO-S12",
            "repo": "https://github.com/zhu-xlab/SSL4EO-S12",
            "num_input_channels": 13,
        },
    )

    def get_state_dict(self) -> Dict[str, Any]:
        """Retrieve pretrained weights state_dict."""
        root = os.path.join(torch.hub.get_dir(), "checkpoints")
        map_location = torch.device("cpu")

        # need to define identifiers for each case

        if "SENTINEL2_ALL_DINO" in str(self):
            filename = "resnet50_dino_sentinel2_all.pth"
            ckpt = load_state_dict_from_url(root, filename, self.url, map_location)
            state_dict = adjust_dino_weights_zhu_lab(ckpt["teacher"])

        elif "SENTINEL1_GRD_MOCO" in str(self):
            filename = "resnet50_dino_sentinel1_grd.pth"
            ckpt = load_state_dict_from_url(root, filename, self.url, map_location)
            state_dict = adjust_moco_weights_zhu_lab(ckpt["state_dict"])

        elif "SENTINEL2_ALL_MOCO" in str(self):
            filename = "resnet50_moco_sentinel2_all.pth"
            ckpt = load_state_dict_from_url(root, filename, self.url, map_location)
            state_dict = adjust_moco_weights_zhu_lab(ckpt["state_dict"])

        elif "SENTINEL2_RGB_MOCO" in str(self):
            filename = "resnet50_moco_sentinel2_rgb.pth"
            ckpt = load_state_dict_from_url(root, filename, self.url, map_location)
            state_dict = adjust_moco_weights_zhu_lab(ckpt["state_dict"])

        elif "IMAGENET_RGB" in str(self):
            filename = "resnet50_imagenet_rgb.pth"
            state_dict = load_state_dict_from_url(
                root, filename, self.url, map_location
            )

        elif "GOOGLEEARTH_MILLIONAID_RGB" in str(self):
            try:
                import yacs  # noqa: F401
            except ImportError:
                raise ImportError(
                    "yacs is not installed but is required to load these weights."
                )
            filename = "resnet50_millionaid_rgb.pth"
            state_dict = load_state_dict_from_url(
                root, filename, self.url, map_location
            )

        return state_dict


def adjust_moco_weights_zhu_lab(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Loading Moco ResNet weights from https://github.com/zhu-xlab/SSL4EO-S12.

    # https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/
    # src/benchmark/transfer_classification/linear_BE_moco.py#L248-L276
    """
    for k in list(state_dict.keys()):
        # retain only encoder up to before the embedding layer
        if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
            # remove prefix
            state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    return state_dict


def adjust_dino_weights_zhu_lab(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """Loading Dino weights from https://github.com/zhu-xlab/SSL4EO-S12.

    # https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/
           # src/benchmark/transfer_classification/models/dino/utils.py#L92-L103
    """
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    # remove `backbone.` prefix induced by multicrop wrapper
    state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
    return state_dict


def load_state_dict_from_url(
    root: str, filename: str, url: str, map_location: torch.device
) -> Dict[str, Any]:
    """Download and load a checkpoint.

    Args:
        root: root directory where checkpoint file should be saved
        filename: filename the downloaded checkpoint file should have
        url: url to checkpoint file to download
        map_location: torch device to load checkpoint

    Returns:
        loaded checkpoint
    """
    if not os.path.exists(os.path.join(root, filename)):
        download_url(url, root=root, filename=filename)
    return torch.load(os.path.join(root, filename), map_location=map_location)
