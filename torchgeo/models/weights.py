"""Pretrained weights for models."""
from torchvision.models._api import Weights

import torchgeo.models.resnet as resnet_weights
import torchgeo.models.vit as vit_weights

BACKBONE_NAMES = ["resnet18", "resnet50", "vit_small_patch16_224"]


def lookup_pretrained_weights(backbone: str, weight_name: str) -> Weights:
    """Lookup correct pretrained weights.

    This is useful when running experiments from a YAML file and
    looking for the desired pretrained weight based on *weight_name*.

    Args:
        backbone: backbone name
        weight_name: pretrained weight name

    Returns:
        Pretrained Weight from WeightEnum

    Raises:
        ValueError if *backbone* does not have pretrained weights available
    """
    if backbone == "resnet18":
        return resnet_weights.ResNet18_Weights[weight_name]
    elif backbone == "resnet50":
        return resnet_weights.ResNet50_Weights[weight_name]
    elif backbone == "vit_small_patch16_224":
        return vit_weights.VITSmall16_Weights[weight_name]
    else:
        raise ValueError(
            f"Currently, only pretrained weights for {BACKBONE_NAMES} are available."
        )
