# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .classification import (
    ClassificationTask,
    MultiLabelClassificationTask,
    So2SatClassificationTask,
)
from .regression import RegressionTask
from .segmentation import (
    ChesapeakeCVPRSegmentationTask,
    LandcoverAISegmentationTask,
    NAIPChesapeakeSegmentationTask,
    SEN12MSSegmentationTask,
)

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
