# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo pre-trained model repository configuration file.

* https://pytorch.org/hub/
* https://pytorch.org/docs/stable/hub.html
"""

from torchgeo.models import (
    copernicusfm_base,
    croma_base,
    croma_large,
    dofa_base_patch16_224,
    dofa_large_patch16_224,
    resnet18,
    resnet50,
    resnet152,
    scalemae_large_patch16,
    swin_v2_b,
    swin_v2_t,
    vit_base_patch16_224,
    vit_huge_patch14_224,
    vit_large_patch16_224,
    vit_small_patch16_224,
)

__all__ = (
    'copernicusfm_base',
    'croma_base',
    'croma_large',
    'dofa_base_patch16_224',
    'dofa_large_patch16_224',
    'resnet18',
    'resnet50',
    'resnet152',
    'scalemae_large_patch16',
    'swin_v2_b',
    'swin_v2_t',
    'vit_base_patch16_224',
    'vit_huge_patch14_224',
    'vit_large_patch16_224',
    'vit_small_patch16_224',
)

dependencies = ['timm', 'torchvision']
