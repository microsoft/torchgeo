# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BigEarthNet trainer."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from ..datasets import BigEarthNet
from .tasks import MultiLabelClassificationTask


class BigEarthNetClassificationTask(MultiLabelClassificationTask):
    """LightningModule for training models on the BigEarthNet Dataset."""

    num_classes = 43
