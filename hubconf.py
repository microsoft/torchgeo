# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo pre-trained model repository configuration file.

* https://pytorch.org/hub/
* https://pytorch.org/docs/stable/hub.html
"""

from torchgeo.models import (
    dofa_base_patch16_224,
    dofa_large_patch16_224,
    resnet18,
    resnet50,
    scalemae_large_patch16,
    swin_v2_b,
    vit_small_patch16_224,
)

__all__ = (
    'dofa_base_patch16_224',
    'dofa_large_patch16_224',
    'resnet18',
    'resnet50',
    'swin_v2_b',
    'scalemae_large_patch16',
    'vit_small_patch16_224',
)

dependencies = ['timm']
