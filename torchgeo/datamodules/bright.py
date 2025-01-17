# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BRIGHT datamodule."""


from typing import Any

import kornia.augmentation as K
import torch

from ..datasets import CaFFe
from .geo import NonGeoDataModule

class BRIGHTDataModule(NonGeoDataModule):
    """LightningDataModule implementation for the BRIGHT_DFC25 dataset.
    
    Implements the default splits that come with the dataset. Note
    that the test split does not have any targets.

    .. versionadded:: 0.7
    """

    # pre image normalization
    # https://github.com/ChenHongruixuan/BRIGHT/blob/11b1ffafa4d30d2df2081189b56864b0de4e3ed7/dfc25_benchmark/dataset/imutils.py#L5
    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.12, 57.375]