# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo: datasets, transforms, and models for geospatial data.

This library is part of the `PyTorch <http://pytorch.org/>`_ project. PyTorch is an open
source machine learning framework.

The :mod:`torchgeo` package consists of popular datasets, model architectures, and
common image transformations for geospatial data.
"""
from typing import Dict, Tuple, Type

import pytorch_lightning as pl

from .datamodules import (
    BigEarthNetDataModule,
    ChesapeakeCVPRDataModule,
    COWCCountingDataModule,
    CycloneDataModule,
    ETCI2021DataModule,
    EuroSATDataModule,
    LandCoverAIDataModule,
    NAIPChesapeakeDataModule,
    OSCDDataModule,
    RESISC45DataModule,
    SEN12MSDataModule,
    So2SatDataModule,
    UCMercedDataModule,
)
from .trainers import (
    BYOLTask,
    ClassificationTask,
    MultiLabelClassificationTask,
    RegressionTask,
    SemanticSegmentationTask,
)
from .trainers.chesapeake import ChesapeakeCVPRSegmentationTask
from .trainers.landcoverai import LandCoverAISegmentationTask
from .trainers.naipchesapeake import NAIPChesapeakeSegmentationTask
from .trainers.resisc45 import RESISC45ClassificationTask

__author__ = "Adam J. Stewart"
__version__ = "0.2.0.dev0"

_TASK_TO_MODULES_MAPPING: Dict[
    str, Tuple[Type[pl.LightningModule], Type[pl.LightningDataModule]]
] = {
    "bigearthnet": (MultiLabelClassificationTask, BigEarthNetDataModule),
    "byol": (BYOLTask, ChesapeakeCVPRDataModule),
    "chesapeake_cvpr": (ChesapeakeCVPRSegmentationTask, ChesapeakeCVPRDataModule),
    "cowc_counting": (RegressionTask, COWCCountingDataModule),
    "cyclone": (RegressionTask, CycloneDataModule),
    "eurosat": (ClassificationTask, EuroSATDataModule),
    "etci2021": (SemanticSegmentationTask, ETCI2021DataModule),
    "landcoverai": (LandCoverAISegmentationTask, LandCoverAIDataModule),
    "naipchesapeake": (NAIPChesapeakeSegmentationTask, NAIPChesapeakeDataModule),
    "oscd": (SemanticSegmentationTask, OSCDDataModule),
    "resisc45": (RESISC45ClassificationTask, RESISC45DataModule),
    "sen12ms": (SemanticSegmentationTask, SEN12MSDataModule),
    "so2sat": (ClassificationTask, So2SatDataModule),
    "ucmerced": (ClassificationTask, UCMercedDataModule),
}
