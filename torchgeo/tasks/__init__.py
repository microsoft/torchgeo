# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .base import BaseTask
from .byol import BYOL
from .change import ChangeDetection
from .classification import Classification
from .detection import ObjectDetection
from .instance_segmentation import InstanceSegmentation
from .iobench import IOBench
from .mae import MAE
from .mixins import ClassificationMixin, RegressionMixin
from .moco import MoCo
from .regression import PixelwiseRegression, Regression
from .segmentation import SemanticSegmentation
from .simclr import SimCLR
from .spatiotemporal_segmentation import SpatioTemporalSegmentation
from .temporal_regression import TemporalRegression

__all__ = (
    'BYOL',
    'MAE',
    'BaseTask',
    'ChangeDetection',
    'Classification',
    'ClassificationMixin',
    'IOBench',
    'InstanceSegmentation',
    'MoCo',
    'ObjectDetection',
    'PixelwiseRegression',
    'Regression',
    'RegressionMixin',
    'SemanticSegmentation',
    'SimCLR',
    'SpatioTemporalSegmentation',
    'TemporalRegression',
)
