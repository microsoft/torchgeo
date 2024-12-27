# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math

import torch
from torch import nn
from torchvision.models._api import Weights


def load_pretrained(
    model: nn.Module,
    weights: Weights,
    pretrained_cfg: dict,
    in_chans: int = 3,
    strict: bool = True,
) -> tuple:
    state_dict = weights.get_state_dict(progress=True)

    input_convs = pretrained_cfg.get('first_conv', None)
    if input_convs is not None:
        if isinstance(input_convs, str):
            input_convs = (input_convs,)
        for input_conv_name in input_convs:
            weight_name = input_conv_name + '.weight'
            weight_in_chans = state_dict[weight_name].shape[1]
            if in_chans != weight_in_chans:
                try:
                    state_dict[weight_name] = adapt_input_conv(
                        in_chans, state_dict[weight_name]
                    )
                    print(
                        f'Converted input conv {input_conv_name} pretrained weights from {weight_in_chans} to {in_chans} channel(s)'
                    )
                except NotImplementedError:
                    del state_dict[weight_name]
                    strict = False
                    print(
                        f'Unable to convert pretrained {input_conv_name} weights, using random init for this layer.'
                    )

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    return missing_keys, unexpected_keys


def adapt_input_conv(in_chans: int, conv_weight: torch.Tensor) -> torch.Tensor:
    conv_type = conv_weight.dtype
    conv_weight = (
        conv_weight.float()
    )  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        conv_weight = conv_weight.sum(dim=1, keepdim=True)
    else:
        repeat = math.ceil(in_chans / I)
        conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
        conv_weight *= I / float(in_chans)
    conv_weight = conv_weight.to(conv_type)
    return conv_weight
