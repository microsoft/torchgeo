# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .base import BaseTask
from .byol import BYOLTask
from .change import ChangeDetectionTask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .detection import ObjectDetectionTask
from .iobench import IOBenchTask
from .moco import MoCoTask
from .regression import PixelwiseRegressionTask, RegressionTask
from .segmentation import SemanticSegmentationTask
from .simclr import SimCLRTask

__all__ = (
    # Supervised
    'BYOLTask',
    'BaseTask',
    'ChangeDetectionTask',
    'ClassificationTask',
    'IOBenchTask',
    'MoCoTask',
    'MultiLabelClassificationTask',
    'ObjectDetectionTask',
    'PixelwiseRegressionTask',
    'RegressionTask',
    'SemanticSegmentationTask',
    'SimCLRTask',
)
