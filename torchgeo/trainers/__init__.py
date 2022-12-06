# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .detection import ObjectDetectionTask
from .regression import RegressionTask
from .segmentation import SemanticSegmentationTask

__all__ = (
    "BYOLTask",
    "ClassificationTask",
    "MultiLabelClassificationTask",
    "ObjectDetectionTask",
    "RegressionTask",
    "SemanticSegmentationTask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
