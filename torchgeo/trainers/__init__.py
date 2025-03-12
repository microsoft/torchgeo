# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .autoregression import AutoregressionTask
from .base import BaseTask
from .byol import BYOLTask
from .classification import ClassificationTask, MultiLabelClassificationTask
from .detection import ObjectDetectionTask
from .instance_segmentation import InstanceSegmentationTask
from .iobench import IOBenchTask
from .moco import MoCoTask
from .regression import PixelwiseRegressionTask, RegressionTask
from .segmentation import SemanticSegmentationTask
from .simclr import SimCLRTask

__all__ = (
    'AutoregressionTask',
    'BYOLTask',
    'BaseTask',
    'ClassificationTask',
    'IOBenchTask',
    'InstanceSegmentationTask',
    'MoCoTask',
    'MultiLabelClassificationTask',
    'ObjectDetectionTask',
    'PixelwiseRegressionTask',
    'RegressionTask',
    'SemanticSegmentationTask',
    'SimCLRTask',
)
