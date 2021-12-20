# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .regression import RegressionTask
from .segmentation import SemanticSegmentationTask

__all__ = (
    "BYOLTask",
    "ClassificationTask",
    "MultiLabelClassificationTask",
    "RegressionTask",
    "SemanticSegmentationTask",
)
