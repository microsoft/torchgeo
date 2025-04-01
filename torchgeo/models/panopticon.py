# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import torch
from torchvision.models._api import Weights, WeightsEnum


class Panopticon_Weights(WeightsEnum):  # type: ignore[misc]
    """Panopticon weights."""

    VIT_BASE14 = Weights(
        url='[TBD]',
        transforms=None,
        meta={
            'model': 'panopticon_vitb14',
            'publication': 'https://arxiv.org/abs/2503.10845',
            'repo': 'https://github.com/Panopticon-FM/panopticon',
            'ssl_method': 'dinov2+spectral_progressive_pretraining',
        },
    )


def panopticon_vitb14() -> torch.nn.Module:
    """Panopticon ViT-Base model.

    Panopticon can handle arbitrary optical channel and SAR combinations.
    For optimal performance, please match the training settings of 224x224 images
    with a patch size of 14. For more information on how to use the model,
    see https://github.com/Panopticon-FM/panopticon?tab=readme-ov-file#using-panopticon.

    If you use this model in your research, please cite the following paper:
    
    * https://arxiv.org/abs/2503.10845
    
    .. versionadded:: 0.7

    Returns:
        The Panopticon ViT-Base model with the published weights loaded.
    """
    return torch.hub.load('Panopticon-FM/panopticon', 'panopticon_vitb14')
