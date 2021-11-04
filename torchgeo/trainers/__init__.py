# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .chesapeake import ChesapeakeCVPRSegmentationTask
from .landcoverai import LandcoverAISegmentationTask
from .naipchesapeake import NAIPChesapeakeSegmentationTask
from .sen12ms import SEN12MSSegmentationTask
from .so2sat import So2SatClassificationTask
from .tasks import ClassificationTask, MultiLabelClassificationTask, RegressionTask

__all__ = (
    "BYOLTask",
    "ChesapeakeCVPRSegmentationTask",
    "ClassificationTask",
    "LandcoverAISegmentationTask",
    "MultiLabelClassificationTask",
    "NAIPChesapeakeSegmentationTask",
    "RegressionTask",
    "SEN12MSSegmentationTask",
    "So2SatClassificationTask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
