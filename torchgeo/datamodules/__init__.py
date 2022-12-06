# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datamodules."""

from .bigearthnet import BigEarthNetDataModule
from .chesapeake import ChesapeakeCVPRDataModule
from .cowc import COWCCountingDataModule
from .cyclone import TropicalCycloneDataModule
from .deepglobelandcover import DeepGlobeLandCoverDataModule
from .etci2021 import ETCI2021DataModule
from .eurosat import EuroSATDataModule
from .fair1m import FAIR1MDataModule
from .inria import InriaAerialImageLabelingDataModule
from .landcoverai import LandCoverAIDataModule
from .loveda import LoveDADataModule
from .naip import NAIPChesapeakeDataModule
from .nasa_marine_debris import NASAMarineDebrisDataModule
from .oscd import OSCDDataModule
from .potsdam import Potsdam2DDataModule
from .resisc45 import RESISC45DataModule
from .sen12ms import SEN12MSDataModule
from .so2sat import So2SatDataModule
from .ucmerced import UCMercedDataModule
from .usavars import USAVarsDataModule
from .vaihingen import Vaihingen2DDataModule
from .xview import XView2DataModule

__all__ = (
    # GeoDataset
    "ChesapeakeCVPRDataModule",
    "NAIPChesapeakeDataModule",
    # NonGeoDataset
    "BigEarthNetDataModule",
    "COWCCountingDataModule",
    "DeepGlobeLandCoverDataModule",
    "ETCI2021DataModule",
    "EuroSATDataModule",
    "FAIR1MDataModule",
    "InriaAerialImageLabelingDataModule",
    "LandCoverAIDataModule",
    "LoveDADataModule",
    "NASAMarineDebrisDataModule",
    "OSCDDataModule",
    "Potsdam2DDataModule",
    "RESISC45DataModule",
    "SEN12MSDataModule",
    "So2SatDataModule",
    "TropicalCycloneDataModule",
    "UCMercedDataModule",
    "USAVarsDataModule",
    "Vaihingen2DDataModule",
    "XView2DataModule",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.datamodules"
