# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .bigearthnet import BigEarthNetClassificationTask
from .byol import BYOLTask
from .chesapeake import ChesapeakeCVPRSegmentationTask
from .landcoverai import LandcoverAISegmentationTask
from .naipchesapeake import NAIPChesapeakeSegmentationTask
from .resisc45 import RESISC45ClassificationTask
from .sen12ms import SEN12MSSegmentationTask
from .so2sat import So2SatClassificationTask
from .tasks import ClassificationTask, MultiLabelClassificationTask, RegressionTask
from .ucmerced import UCMercedClassificationTask

__all__ = (
    "BigEarthNetClassificationTask",
    "BYOLTask",
    "ChesapeakeCVPRSegmentationTask",
    "ClassificationTask",
    "LandcoverAISegmentationTask",
    "MultiLabelClassificationTask",
    "NAIPChesapeakeSegmentationTask",
    "RESISC45ClassificationTask",
    "RegressionTask",
    "SEN12MSSegmentationTask",
    "So2SatClassificationTask",
    "UCMercedClassificationTask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
