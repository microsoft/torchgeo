# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common trainer utilities."""

import warnings
from collections import OrderedDict
from typing import Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Conv2d, Module

# https://github.com/pytorch/pytorch/issues/60979
# https://github.com/pytorch/pytorch/pull/61045
Module.__module__ = "nn.Module"
Conv2d.__module__ = "nn.Conv2d"


def extract_encoder(path: str) -> Tuple[str, "OrderedDict[str, Tensor]"]:
    """Extracts an encoder from a pytorch lightning checkpoint file.

    Args:
        path: path to checkpoint file (.ckpt)

    Returns:
        tuple containing model name and state dict

    Raises:
        ValueError: if 'model' or 'encoder' not in
            checkpoint['hyper_parameters']
    """
    checkpoint = torch.load(path, map_location=torch.device("cpu"))
    if "model" in checkpoint["hyper_parameters"]:
        name = checkpoint["hyper_parameters"]["model"]
        state_dict = checkpoint["state_dict"]
        state_dict = OrderedDict({k: v for k, v in state_dict.items() if "model." in k})
        state_dict = OrderedDict(
            {k.replace("model.", ""): v for k, v in state_dict.items()}
        )
    elif "encoder_name" in checkpoint["hyper_parameters"]:
        name = checkpoint["hyper_parameters"]["encoder_name"]
        state_dict = checkpoint["state_dict"]
        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if "model.encoder.model" in k}
        )
        state_dict = OrderedDict(
            {k.replace("model.encoder.model.", ""): v for k, v in state_dict.items()}
        )
    else:
        raise ValueError(
            "Unknown checkpoint task. Only encoder or model extraction is supported"
        )

    return name, state_dict


def load_state_dict(model: Module, state_dict: "OrderedDict[str, Tensor]") -> Module:
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
    in_channels = cast(nn.Module, model.conv1).in_channels
    expected_in_channels = state_dict["conv1.weight"].shape[1]
    num_classes = cast(nn.Module, model.fc).out_features
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

    model.load_state_dict(state_dict, strict=False)

    return model


def reinit_initial_conv_layer(
    layer: Conv2d,
    new_in_channels: int,
    keep_rgb_weights: bool,
    new_stride: Optional[Union[int, Tuple[int, int]]] = None,
    new_padding: Optional[Union[str, Union[int, Tuple[int, int]]]] = None,
) -> Conv2d:
    """Clones a Conv2d layer while optionally retaining some of the original weights.

    When replacing the first convolutional layer in a model with one that operates over
    different number of input channels, we sometimes want to keep a subset of the kernel
    weights the same (e.g. the RGB weights of an ImageNet pretrained model). This is a
    convenience function that performs that function.

    Args:
        layer: the Conv2d layer to initialize
        new_in_channels: the new number of input channels
        keep_rgb_weights: flag indicating whether to re-initialize the first 3 channels
        new_stride: optionally, overwrites the ``layer``'s stride with this value
        new_padding: optionally, overwrites the ``layers``'s padding with this value

    Returns:
        a Conv2d layer with new kernel weights
    """
    use_bias = layer.bias is not None
    if keep_rgb_weights:
        w_old = layer.weight.data[:, :3, :, :].clone()
        if use_bias:
            b_old = cast(Tensor, layer.bias).data.clone()

    updated_stride = layer.stride if new_stride is None else new_stride
    updated_padding = layer.padding if new_padding is None else new_padding

    new_layer = Conv2d(
        new_in_channels,
        layer.out_channels,
        kernel_size=layer.kernel_size,  # type: ignore[arg-type]
        stride=updated_stride,  # type: ignore[arg-type]
        padding=updated_padding,  # type: ignore[arg-type]
        dilation=layer.dilation,  # type: ignore[arg-type]
        groups=layer.groups,
        bias=use_bias,
        padding_mode=layer.padding_mode,
    )
    nn.init.kaiming_normal_(new_layer.weight, mode="fan_out", nonlinearity="relu")

    if keep_rgb_weights:
        new_layer.weight.data[:, :3, :, :] = w_old
        if use_bias:
            cast(Tensor, new_layer.bias).data = b_old

    return new_layer
