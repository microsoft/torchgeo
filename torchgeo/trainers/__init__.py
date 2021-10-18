# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .byol import BYOLTask
from .chesapeake import ChesapeakeCVPRDataModule, ChesapeakeCVPRSegmentationTask
from .chesapeake_learn_on_prior import ChesapeakeCVPRPriorDataModule, ChesapeakeCVPRPriorSegmentationTask
from .cyclone import CycloneDataModule, CycloneSimpleRegressionTask
from .landcoverai import LandcoverAIDataModule, LandcoverAISegmentationTask
from .naipchesapeake import NAIPChesapeakeDataModule, NAIPChesapeakeSegmentationTask
from .resisc45 import RESISC45ClassificationTask, RESISC45DataModule
from .sen12ms import SEN12MSDataModule, SEN12MSSegmentationTask
from .so2sat import So2SatClassificationTask, So2SatDataModule
from .tasks import ClassificationTask
from .ucmerced import UCMercedClassificationTask, UCMercedDataModule

__all__ = (
    # Tasks
    "ClassificationTask",
    # Trainers
    "BYOLTask",
    "ChesapeakeCVPRSegmentationTask",
    "ChesapeakeCVPRDataModule",
    "ChesapeakeCVPRPriorSegmentationTask",
    "ChesapeakeCVPRPriorDataModule",
    "CycloneDataModule",
    "CycloneSimpleRegressionTask",
    "LandcoverAIDataModule",
    "LandcoverAISegmentationTask",
    "NAIPChesapeakeDataModule",
    "NAIPChesapeakeSegmentationTask",
    "RESISC45ClassificationTask",
    "RESISC45DataModule",
    "SEN12MSDataModule",
    "SEN12MSSegmentationTask",
    "So2SatDataModule",
    "So2SatClassificationTask",
    "UCMercedClassificationTask",
    "UCMercedDataModule",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
