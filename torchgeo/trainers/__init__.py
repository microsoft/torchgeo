# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .detection import ObjectDetectionTask
from .moco import MoCoTask
from .regression import PixelwiseRegressionTask, RegressionTask
from .segmentation import SemanticSegmentationTask
from .simclr import SimCLRTask

__all__ = (
    "BYOLTask",
    "ClassificationTask",
    "MoCoTask",
    "MultiLabelClassificationTask",
    "ObjectDetectionTask",
    "PixelwiseRegressionTask",
    "RegressionTask",
    "SemanticSegmentationTask",
    "SimCLRTask",
)
