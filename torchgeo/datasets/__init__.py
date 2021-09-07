# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datasets."""

from .benin_cashews import BeninSmallHolderCashews
from .cbf import CanadianBuildingFootprints
from .cdl import CDL
from .chesapeake import (
    Chesapeake,
    Chesapeake7,
    Chesapeake13,
    ChesapeakeCVPR,
    ChesapeakeDC,
    ChesapeakeDE,
    ChesapeakeMD,
    ChesapeakeNY,
    ChesapeakePA,
    ChesapeakeVA,
    ChesapeakeWV,
)
from .cowc import COWC, COWCCounting, COWCDetection
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import TropicalCycloneWindEstimation
from .geo import GeoDataset, RasterDataset, VectorDataset, VisionDataset, ZipDataset
from .landcoverai import LandCoverAI
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
from .naip import NAIP
from .nwpu import VHR10
from .patternnet import PatternNet
from .sen12ms import SEN12MS
from .sentinel import Sentinel, Sentinel2
from .so2sat import So2Sat
from .utils import BoundingBox, collate_dict

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
    "Sentinel",
    "Sentinel2",
    # VisionDataset
    "BeninSmallHolderCashews",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "CV4AKenyaCropType",
    "LandCoverAI",
    "LEVIRCDPlus",
    "PatternNet",
    "SEN12MS",
    "So2Sat",
    "TropicalCycloneWindEstimation",
    "VHR10",
    # Base classes
    "GeoDataset",
    "RasterDataset",
    "VectorDataset",
    "VisionDataset",
    "ZipDataset",
    # Utilities
    "BoundingBox",
    "collate_dict",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.datasets"
