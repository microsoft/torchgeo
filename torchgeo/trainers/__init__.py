# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .chesapeake import ChesapeakeCVPRSegmentationTask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .etci2021 import ETCI2021SemanticSegmentationTask
from .landcoverai import LandCoverAISegmentationTask
from .naipchesapeake import NAIPChesapeakeSegmentationTask
from .regression import RegressionTask
from .resisc45 import RESISC45ClassificationTask
from .segmentation import SemanticSegmentationTask
from .so2sat import So2SatClassificationTask

__all__ = (
    "BYOLTask",
    "ChesapeakeCVPRSegmentationTask",
    "ClassificationTask",
    "ETCI2021SemanticSegmentationTask",
    "LandCoverAISegmentationTask",
    "MultiLabelClassificationTask",
    "NAIPChesapeakeSegmentationTask",
    "RegressionTask",
    "RESISC45ClassificationTask",
    "SemanticSegmentationTask",
    "So2SatClassificationTask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
