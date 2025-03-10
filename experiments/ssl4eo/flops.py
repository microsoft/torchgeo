#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import timm
from deepspeed.accelerator import get_accelerator
from deepspeed.profiling.flops_profiler import get_model_profile

models = ['resnet18', 'resnet50', 'vit_small_patch16_224']
num_classes = 14
in_channels = 11
batch_size = 64
patch_size = 224
input_shape = (batch_size, in_channels, patch_size, patch_size)

for model in models:
    print(f'Model: {model}')

    m = timm.create_model(model, num_classes=num_classes, in_chans=in_channels)

    # Calculate memory requirements of model
    mem_params = sum([p.nelement() * p.element_size() for p in m.parameters()])
    mem_bufs = sum([b.nelement() * b.element_size() for b in m.buffers()])
    mem = (mem_params + mem_bufs) / 1000000
    print(f'Memory: {mem:.2f} MB')

    with get_accelerator().device(0):
        get_model_profile(
            model=m, input_shape=input_shape, detailed=False, module_depth=0
        )
