# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common trainer utilities."""

import warnings
from collections import OrderedDict
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn.modules import Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "nn.Module"


def extract_encoder(path: str) -> Tuple[str, Dict[str, Tensor]]:
    """Extracts an encoder from a pytorch lightning checkpoint file.

    Args:
        path: path to checkpoint file (.ckpt)

    Returns:
        tuple containing model name and state dict

    Raises:
        ValueError: if 'classification_model' or 'encoder' not in
            checkpoint['hyper_parameters']
    """
    checkpoint = torch.load(  # type: ignore[no-untyped-call]
        path, map_location=torch.device("cpu")  # type: ignore[attr-defined]
    )

    if "classification_model" in checkpoint["hyper_parameters"]:
        name = checkpoint["hyper_parameters"]["classification_model"]
        state_dict = checkpoint["state_dict"]
        state_dict = OrderedDict({k: v for k, v in state_dict.items() if "model." in k})
        state_dict = OrderedDict(
            {k.replace("model.", ""): v for k, v in state_dict.items()}
        )
    elif "encoder" in checkpoint["hyper_parameters"]:
        name = checkpoint["hyper_parameters"]["encoder"]
        state_dict = checkpoint["state_dict"]
        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if "model.encoder.model" in k}
        )
        state_dict = OrderedDict(
            {k.replace("model.encoder.model.", ""): v for k, v in state_dict.items()}
        )
    else:
        raise ValueError(
            "Unknown checkpoint task. Only encoder or classification_model"
            " extraction is supported"
        )

    return name, state_dict


def load_state_dict(model: Module, state_dict: Dict[str, Tensor]) -> Module:
    """Load pretrained resnet weights to a model.

    Args:
        model: model to load the pretrained weights to
        state_dict: dict containing tensor parameters

    Returns:
        the model with pretrained weights

    Warns:
        If input channels in model != pretrained model input channels
        If num output classes in model != pretrained model num classes
    """
    in_channels = model.conv1.in_channels  # type: ignore[union-attr]
    expected_in_channels = state_dict["conv1.weight"].shape[1]
    num_classes = model.fc.out_features  # type: ignore[union-attr]
    expected_num_classes = state_dict["fc.weight"].shape[0]

    if in_channels != expected_in_channels:
        warnings.warn(
            f"input channels {in_channels} != input channels in pretrained"
            f" model {expected_in_channels}. Overriding with new input channels"
        )
        del state_dict["conv1.weight"]

    if num_classes != expected_num_classes:
        warnings.warn(
            f"num classes {num_classes} != num classes in pretrained model"
            f" {expected_num_classes}. Overriding with new num classes"
        )
        del state_dict["fc.weight"], state_dict["fc.bias"]

    model.load_state_dict(state_dict, strict=False)  # type: ignore[arg-type]

    return model
