# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datasets."""

from .advance import ADVANCE
from .benin_cashews import BeninSmallHolderCashews
from .bigearthnet import BigEarthNet, BigEarthNetDataModule
from .cbf import CanadianBuildingFootprints
from .cdl import CDL
from .chesapeake import (
    Chesapeake,
    Chesapeake7,
    Chesapeake13,
    ChesapeakeCVPR,
    ChesapeakeCVPRDataModule,
    ChesapeakeDC,
    ChesapeakeDE,
    ChesapeakeMD,
    ChesapeakeNY,
    ChesapeakePA,
    ChesapeakeVA,
    ChesapeakeWV,
)
from .cowc import COWC, COWCCounting, COWCCountingDataModule, COWCDetection
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import CycloneDataModule, TropicalCycloneWindEstimation
from .etci2021 import ETCI2021
from .eurosat import EuroSAT
from .geo import (
    GeoDataset,
    RasterDataset,
    VectorDataset,
    VisionClassificationDataset,
    VisionDataset,
    ZipDataset,
)
from .gid15 import GID15
from .landcoverai import LandCoverAI, LandCoverAIDataModule
from .landsat import (
    Landsat,
    Landsat1,
    Landsat2,
    Landsat3,
    Landsat4MSS,
    Landsat4TM,
    Landsat5MSS,
    Landsat5TM,
    Landsat7,
    Landsat8,
    Landsat9,
)
from .levircd import LEVIRCDPlus
from .naip import NAIP, NAIPChesapeakeDataModule
from .nwpu import VHR10
from .patternnet import PatternNet
from .resisc45 import RESISC45, RESISC45DataModule
from .seco import SeasonalContrastS2
from .sen12ms import SEN12MS, SEN12MSDataModule
from .sentinel import Sentinel, Sentinel2
from .so2sat import So2Sat, So2SatDataModule
from .spacenet import SpaceNet, SpaceNet1, SpaceNet2, SpaceNet4
from .ucmerced import UCMerced, UCMercedDataModule
from .utils import BoundingBox, collate_dict
from .zuericrop import ZueriCrop

__all__ = (
    # GeoDataset
    "CanadianBuildingFootprints",
    "CDL",
    "Chesapeake",
    "Chesapeake7",
    "Chesapeake13",
    "ChesapeakeDC",
    "ChesapeakeDE",
    "ChesapeakeMD",
    "ChesapeakeNY",
    "ChesapeakePA",
    "ChesapeakeVA",
    "ChesapeakeWV",
    "ChesapeakeCVPR",
    "ChesapeakeCVPRDataModule",
    "Landsat",
    "Landsat1",
    "Landsat2",
    "Landsat3",
    "Landsat4MSS",
    "Landsat4TM",
    "Landsat5MSS",
    "Landsat5TM",
    "Landsat7",
    "Landsat8",
    "Landsat9",
    "NAIP",
    "NAIPChesapeakeDataModule",
    "Sentinel",
    "Sentinel2",
    # VisionDataset
    "ADVANCE",
    "BeninSmallHolderCashews",
    "BigEarthNet",
    "BigEarthNetDataModule",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "COWCCountingDataModule",
    "CV4AKenyaCropType",
    "ETCI2021",
    "EuroSAT",
    "GID15",
    "LandCoverAI",
    "LandCoverAIDataModule",
    "LEVIRCDPlus",
    "PatternNet",
    "RESISC45",
    "RESISC45DataModule",
    "SeasonalContrastS2",
    "SEN12MS",
    "SEN12MSDataModule",
    "So2Sat",
    "So2SatDataModule",
    "SpaceNet",
    "SpaceNet1",
    "SpaceNet2",
    "SpaceNet4",
    "TropicalCycloneWindEstimation",
    "CycloneDataModule",
    "UCMerced",
    "UCMercedDataModule",
    "VHR10",
    "ZueriCrop",
    # Base classes
    "GeoDataset",
    "RasterDataset",
    "VectorDataset",
    "VisionDataset",
    "VisionClassificationDataset",
    "ZipDataset",
    # Utilities
    "BoundingBox",
    "collate_dict",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.datasets"
