# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo trainers."""

from .chesapeake import ChesapeakeCVPRDataModule, ChesapeakeCVPRSegmentationTask
from .cyclone import CycloneDataModule, CycloneSimpleRegressionTask
from .landcoverai import LandcoverAIDataModule, LandcoverAISegmentationTask
from .naipchesapeake import NAIPChesapeakeDataModule, NAIPChesapeakeSegmentationTask
from .sen12ms import SEN12MSDataModule, SEN12MSSegmentationTask
from .so2sat import So2SatDataModule, So2SatClassificationTask
from .byol import  BYOLTask

__all__ = (
    "BYOLTask",
    "ChesapeakeCVPRSegmentationTask",
    "ChesapeakeCVPRDataModule",
    "CycloneDataModule",
    "CycloneSimpleRegressionTask",
    "LandcoverAIDataModule",
    "LandcoverAISegmentationTask",
    "NAIPChesapeakeDataModule",
    "NAIPChesapeakeSegmentationTask",
    "SEN12MSDataModule",
    "SEN12MSSegmentationTask",
    "So2SatDataModule",
    "So2SatClassificationTask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
