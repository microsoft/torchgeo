# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datasets."""

from .advance import ADVANCE
from .benin_cashews import BeninSmallHolderCashews
from .bigearthnet import BigEarthNet
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
from .etci2021 import ETCI2021
from .eurosat import EuroSAT
from .fair1m import FAIR1M
from .geo import (
    GeoDataset,
    IntersectionDataset,
    RasterDataset,
    UnionDataset,
    VectorDataset,
    VisionClassificationDataset,
    VisionDataset,
)
from .gid15 import GID15
from .idtrees import IDTReeS
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
from .loveda import LoveDA
from .naip import NAIP
from .nasa_marine_debris import NASAMarineDebris
from .nwpu import VHR10
from .oscd import OSCD
from .patternnet import PatternNet
from .potsdam import Potsdam2D
from .resisc45 import RESISC45
from .seco import SeasonalContrastS2
from .sen12ms import SEN12MS
from .sentinel import Sentinel, Sentinel2
from .so2sat import So2Sat
from .spacenet import SpaceNet, SpaceNet1, SpaceNet2, SpaceNet4, SpaceNet5, SpaceNet7
from .ucmerced import UCMerced
from .utils import (
    BoundingBox,
    concat_samples,
    merge_samples,
    stack_samples,
    unbind_samples,
)
from .vaihingen import Vaihingen2D
from .xview import XView2
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
    "ADVANCE",
    "BeninSmallHolderCashews",
    "BigEarthNet",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "CV4AKenyaCropType",
    "ETCI2021",
    "EuroSAT",
    "FAIR1M",
    "GID15",
    "IDTReeS",
    "LandCoverAI",
    "LEVIRCDPlus",
    "LoveDA",
    "NASAMarineDebris",
    "OSCD",
    "PatternNet",
    "Potsdam2D",
    "RESISC45",
    "SeasonalContrastS2",
    "SEN12MS",
    "So2Sat",
    "SpaceNet",
    "SpaceNet1",
    "SpaceNet2",
    "SpaceNet4",
    "SpaceNet5",
    "SpaceNet7",
    "TropicalCycloneWindEstimation",
    "UCMerced",
    "Vaihingen2D",
    "VHR10",
    "XView2",
    "ZueriCrop",
    # Base classes
    "GeoDataset",
    "IntersectionDataset",
    "RasterDataset",
    "UnionDataset",
    "VectorDataset",
    "VisionDataset",
    "VisionClassificationDataset",
    # Utilities
    "BoundingBox",
    "concat_samples",
    "merge_samples",
    "stack_samples",
    "unbind_samples",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.datasets"
