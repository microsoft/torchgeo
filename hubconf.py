# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo pre-trained model repository configuration file.

* https://pytorch.org/hub/
* https://pytorch.org/docs/stable/hub.html
"""

from torchgeo.models import resnet18, resnet50, vit_small_patch16_224

__all__ = ('resnet18', 'resnet50', 'vit_small_patch16_224')

dependencies = ['timm']
