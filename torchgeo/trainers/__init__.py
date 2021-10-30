# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .bigearthnet import BigEarthNetClassificationTask, BigEarthNetDataModule
from .byol import BYOLTask
from .chesapeake import ChesapeakeCVPRDataModule, ChesapeakeCVPRSegmentationTask
from .cyclone import CycloneDataModule, CycloneSimpleRegressionTask
from .landcoverai import LandcoverAIDataModule, LandcoverAISegmentationTask
from .naipchesapeake import NAIPChesapeakeDataModule, NAIPChesapeakeSegmentationTask
from .resisc45 import RESISC45ClassificationTask, RESISC45DataModule
from .sen12ms import SEN12MSDataModule, SEN12MSSegmentationTask
from .so2sat import So2SatClassificationTask, So2SatDataModule
from .tasks import ClassificationTask, MultiLabelClassificationTask
from .ucmerced import UCMercedClassificationTask, UCMercedDataModule

__all__ = (
    # Base Classes
    "ClassificationTask",
    "MultiLabelClassificationTask",
    # Tasks
    "BigEarthNetClassificationTask",
    "BYOLTask",
    "ChesapeakeCVPRSegmentationTask",
    "CycloneSimpleRegressionTask",
    "LandcoverAISegmentationTask",
    "NAIPChesapeakeSegmentationTask",
    "RESISC45ClassificationTask",
    "SEN12MSSegmentationTask",
    "So2SatClassificationTask",
    "UCMercedClassificationTask",
    # DataModules
    "BigEarthNetDataModule",
    "ChesapeakeCVPRDataModule",
    "CycloneDataModule",
    "LandcoverAIDataModule",
    "NAIPChesapeakeDataModule",
    "RESISC45DataModule",
    "SEN12MSDataModule",
    "So2SatDataModule",
    "UCMercedDataModule",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
