"""Pretrained weights for models."""
import os
from typing import Any, Dict

import torch
import torch.nn as nn
from torchvision.models._api import Weights, WeightsEnum

from torchgeo.datasets.utils import download_url

__all__ = ["ResNet50_Weights"]


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
        },
    )

    MILLIONAID_RGB = Weights(
        url="https://drive.google.com/file/d/1K3P4_fDfcBRGqpKoSdSa6OXS4xC1xLC9",
        transforms=nn.Identity(),
        meta={
            "dataset": "millionaid",
            "ssl_method": None,
            "repo": (
                "https://github.com/ViTAE-Transformer/"
                "ViTAE-Transformer-Remote-Sensing"
            ),
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
        },
    )
    DEFAULT = SENTINEL2_ALL_MOCO

    def get_state_dict(self) -> Dict[str, Any]:
        """Retrieve pretrained weights state_dict."""
        root = os.path.join(os.getcwd(), "weights")
        map_location = torch.device("cpu")

        # need to define identifiers for each case
        if self.meta["ssl_method"] == "dino":
            filename = "resnet50_dino_sentinel_weights.pth"
            ckpt = load_checkpoint_from_url(root, filename, self.url, map_location)
            # https://github.com/zhu-xlab/SSL4EO-S12/blob/1a668f76fd46762a19780293675a6e23e5204e72/
            # src/benchmark/transfer_classification/models/dino/utils.py#L92-L103
            state_dict = ckpt["teacher"]
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

        elif self.meta["ssl_method"] == "moco":
            filename = "resnet50_moco_sentinel_weights.pth"
            ckpt = load_checkpoint_from_url(root, filename, self.url, map_location)
            # https://github.com/zhu-xlab/SSL4EO-S12/blob/d2868adfada65e40910bfcedfc49bc3b20df2248/
            # src/benchmark/transfer_classification/linear_BE_moco.py#L248-L276
            state_dict = ckpt["state_dict"]
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if k.startswith("module.encoder_q") and not k.startswith(
                    "module.encoder_q.fc"
                ):
                    # remove prefix
                    state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
                # delete renamed or unused k
                del state_dict[k]

        elif self.meta["dataset"] == "imagenet":
            filename = "resnet50_imagenet_rgb_weights.pth"
            state_dict = load_checkpoint_from_url(
                root, filename, self.url, map_location
            )

        elif self.meta["dataset"] == "millionaid":
            try:
                import yacs  # noqa: F401
            except ImportError:
                raise ImportError(
                    "yacs is not installed but is required to load these weights."
                )
            filename = "resnet50_millionaid_rgb_weights.pth"
            state_dict = load_checkpoint_from_url(
                root, filename, self.url, map_location
            )

        return state_dict


def load_checkpoint_from_url(
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
