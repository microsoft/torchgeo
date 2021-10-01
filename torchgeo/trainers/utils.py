# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common trainer utilities."""

import warnings
from collections import OrderedDict
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "nn.Module"


def extract_encoder(path: str) -> Tuple[str, OrderedDict[str, Tensor]]:
    """Extracts an encoder from a pytorch lightning checkpoint file.

    Args:
        path: path to checkpoint file (.ckpt)
        output_path: output path to save encoder state dict (.pt)
    """
    checkpoint = torch.load(  # type: ignore[no-untyped-call]
        path, map_location=torch.device("cpu")  # type: ignore[attr-defined]
    )
    _ = checkpoint["hyper_parameters"]["model"]
    name = checkpoint["hyper_parameters"]["encoder"]

    state_dict = checkpoint["state_dict"]
    state_dict = OrderedDict(
        {k: v for k, v in state_dict.items() if "model.encoder.model" in k}
    )
    state_dict = OrderedDict(
        {k.replace("model.encoder.model.", ""): v for k, v in state_dict.items()}
    )
    return name, state_dict


def load_state_dict(model: Module, state_dict: OrderedDict[str, Tensor]) -> Module:
    """Load pretrained model weights to a model.

    Args:
        model: model to load the pretrained weights to
        state_dict: dict containing tensor parameters

    Returns:
        the model with pretrained weights

    Raises:
        Warning: if num output classes in model != pretrained model num classes
    """
    in_channels = state_dict["conv1.weight"].shape[1]
    model.conv1 = nn.modules.Conv2d(
        in_channels,
        64,
        kernel_size=7,
        stride=1,
        padding=2,
        bias=False,
    )
    num_classes = model.fc.out_features  # type: ignore[union-attr]
    expected_num_classes = state_dict["fc.weight"].shape[0]

    if num_classes == expected_num_classes:
        model.load_state_dict(state_dict)
    else:
        warnings.warn(
            f"""num classes {num_classes} != num classes in pretrained model"""
            """{expected_num_classes}. Overriding with new num classes"""
        )
        del state_dict["fc.weight"], state_dict["fc.bias"]
        _ = model.load_state_dict(state_dict, strict=False)

    return model
