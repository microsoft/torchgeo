# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""Common trainer utilities."""

import warnings
from collections import OrderedDict
from typing import cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules import Conv2d, Module
from torchvision.models.detection.transform import GeneralizedRCNNTransform


class GeneralizedRCNNTransformNoOp(GeneralizedRCNNTransform):  # type: ignore[misc]
    """GeneralizedRCNNTransform without the normalize and resize ops.

    .. versionadded:: 0.8
    """

    def __init__(self) -> None:
        """Initialize a new GeneralizedRCNNTransformNoOp instance."""
        super().__init__(min_size=0, max_size=0, image_mean=[0], image_std=[1])

    def resize(
        self, image: Tensor, target: dict[str, Tensor] | None = None
    ) -> tuple[Tensor, dict[str, Tensor] | None]:
        """Skip resizing and return the image and target."""
        return image, target


def extract_backbone(path: str) -> tuple[str, 'OrderedDict[str, Tensor]']:
    """Extracts a backbone from a lightning checkpoint file.

    Args:
        path: path to checkpoint file (.ckpt)

    Returns:
        tuple containing model name and state dict

    Raises:
        ValueError: if 'model' or 'backbone' not in
            checkpoint['hyper_parameters']

    .. versionchanged:: 0.4
        Renamed from *extract_encoder* to *extract_backbone*
    """
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    if 'model' in checkpoint['hyper_parameters']:
        name = checkpoint['hyper_parameters']['model']
        state_dict = checkpoint['state_dict']
        state_dict = OrderedDict({k: v for k, v in state_dict.items() if 'model.' in k})
        state_dict = OrderedDict(
            {k.replace('model.', ''): v for k, v in state_dict.items()}
        )
    elif 'backbone' in checkpoint['hyper_parameters']:
        name = checkpoint['hyper_parameters']['backbone']
        state_dict = checkpoint['state_dict']
        state_dict = OrderedDict(
            {k: v for k, v in state_dict.items() if 'model.backbone.model' in k}
        )
        state_dict = OrderedDict(
            {k.replace('model.backbone.model.', ''): v for k, v in state_dict.items()}
        )
    else:
        raise ValueError(
            'Unknown checkpoint task. Only backbone or model extraction is supported'
        )

    return name, state_dict


def _get_input_layer_name_and_module(model: Module) -> tuple[str, Module]:
    """Retrieve the input layer name and module from a timm model.

    Args:
        model: timm model
    """
    keys = []
    children = list(model.named_children())
    while children != []:
        name, module = children[0]
        keys.append(name)
        children = list(module.named_children())

    key = '.'.join(keys)
    return key, module


def load_state_dict(
    model: Module, state_dict: 'OrderedDict[str, Tensor]'
) -> tuple[list[str], list[str]]:
    """Load pretrained resnet weights to a model.

    Args:
        model: model to load the pretrained weights to
        state_dict: dict containing tensor parameters

    Returns:
        The missing and unexpected keys

    Warns:
        If input channels in model != pretrained model input channels
        If num output classes in model != pretrained model num classes
    """
    input_module_key, input_module = _get_input_layer_name_and_module(model)
    in_channels = input_module.in_channels
    expected_in_channels = state_dict[input_module_key + '.weight'].shape[1]

    output_module_key, output_module = list(model.named_children())[-1]
    if isinstance(output_module, nn.Identity):
        num_classes = model.num_features
    else:
        num_classes = output_module.out_features
    expected_num_classes = None
    if output_module_key + '.weight' in state_dict:
        expected_num_classes = state_dict[output_module_key + '.weight'].shape[0]

    if in_channels != expected_in_channels:
        warnings.warn(
            f'input channels {in_channels} != input channels in pretrained'
            f' model {expected_in_channels}. Overriding with new input channels'
        )
        del state_dict[input_module_key + '.weight']

    if expected_num_classes and num_classes != expected_num_classes:
        warnings.warn(
            f'num classes {num_classes} != num classes in pretrained model'
            f' {expected_num_classes}. Overriding with new num classes'
        )
        del (
            state_dict[output_module_key + '.weight'],
            state_dict[output_module_key + '.bias'],
        )

    missing_keys: list[str]
    unexpected_keys: list[str]
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    return missing_keys, unexpected_keys


def reinit_initial_conv_layer(
    layer: Conv2d,
    new_in_channels: int,
    keep_rgb_weights: bool,
    new_stride: int | tuple[int, int] | None = None,
    new_padding: str | int | tuple[int, int] | None = None,
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
    nn.init.kaiming_normal_(new_layer.weight, mode='fan_out', nonlinearity='relu')

    if keep_rgb_weights:
        new_layer.weight.data[:, :3, :, :] = w_old
        if use_bias:
            cast(Tensor, new_layer.bias).data = b_old

    return new_layer
