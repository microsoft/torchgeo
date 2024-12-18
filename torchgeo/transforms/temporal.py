# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo temporal transforms."""

from typing import Any

import kornia.augmentation as K
import torch
from einops import rearrange
from kornia.contrib import extract_tensor_patches
from kornia.geometry import crop_by_indices
from kornia.geometry.boxes import Boxes
from torch import Tensor
from torch.nn.modules import Module