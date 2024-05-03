# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""APIs for querying and loading pre-trained model weights.

See the following references for design details:

* https://pytorch.org/blog/easily-list-and-initialize-models-with-new-apis-in-torchvision/
* https://pytorch.org/vision/stable/models.html
* https://github.com/pytorch/vision/blob/main/torchvision/models/_api.py
"""  # noqa: E501

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torchvision.models._api import WeightsEnum

from .resnet import ResNet18_Weights, ResNet50_Weights, resnet18, resnet50
from .swin import Swin_V2_B_Weights, swin_v2_b
from .vit import ViTSmall16_Weights, vit_small_patch16_224

_model = {
    'resnet18': resnet18,
    'resnet50': resnet50,
    'vit_small_patch16_224': vit_small_patch16_224,
    'swin_v2_b': swin_v2_b,
}

_model_weights = {
    resnet18: ResNet18_Weights,
    resnet50: ResNet50_Weights,
    vit_small_patch16_224: ViTSmall16_Weights,
    swin_v2_b: Swin_V2_B_Weights,
    'resnet18': ResNet18_Weights,
    'resnet50': ResNet50_Weights,
    'vit_small_patch16_224': ViTSmall16_Weights,
    'swin_v2_b': Swin_V2_B_Weights,
}


def get_model(name: str, *args: Any, **kwargs: Any) -> nn.Module:
    """Get an instantiated model from its name.

    .. versionadded:: 0.4

    Args:
        name: Name of the model.
        *args: Additional arguments passed to the model builder method.
        **kwargs: Additional keyword arguments passed to the model builder method.

    Returns:
        An instantiated model.
    """
    model: nn.Module = _model[name](*args, **kwargs)
    return model


def get_model_weights(name: Callable[..., nn.Module] | str) -> WeightsEnum:
    """Get the weights enum class associated with a given model.

    .. versionadded:: 0.4

    Args:
        name: Model builder function or the name under which it is registered.

    Returns:
        The weights enum class associated with the model.
    """
    return _model_weights[name]


def get_weight(name: str) -> WeightsEnum:
    """Get the weights enum value by its full name.

    .. versionadded:: 0.4

    Args:
        name: Name of the weight enum entry.

    Returns:
        The requested weight enum.
    """
    return eval(name)


def list_models() -> list[str]:
    """List the registered models.

    .. versionadded:: 0.4

    Returns:
        A list of registered models.
    """
    return list(_model.keys())
