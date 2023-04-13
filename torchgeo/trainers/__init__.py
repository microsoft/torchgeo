# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .detection import ObjectDetectionTask
from .regression import DenseRegressionTask, RegressionTask
from .segmentation import SemanticSegmentationTask

__all__ = (
    "BYOLTask",
    "ClassificationTask",
    "DenseRegressionTask",
    "MultiLabelClassificationTask",
    "ObjectDetectionTask",
    "RegressionTask",
    "SemanticSegmentationTask",
)
