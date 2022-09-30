# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datasets."""

from .advance import ADVANCE
from .agb_live_woody_density import AbovegroundLiveWoodyBiomassDensity
from .astergdem import AsterGDEM
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
from .cloud_cover import CloudCoverDetection
from .cms_mangrove_canopy import CMSGlobalMangroveCanopy
from .cowc import COWC, COWCCounting, COWCDetection
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import TropicalCycloneWindEstimation
from .deepglobelandcover import DeepGlobeLandCover
from .dfc2022 import DFC2022
from .eddmaps import EDDMapS
from .enviroatlas import EnviroAtlas
from .esri2020 import Esri2020
from .etci2021 import ETCI2021
from .eudem import EUDEM
from .eurosat import EuroSAT
from .fair1m import FAIR1M
from .forestdamage import ForestDamage
from .gbif import GBIF
from .geo import (
    GeoDataset,
    IntersectionDataset,
    NonGeoClassificationDataset,
    NonGeoDataset,
    RasterDataset,
    UnionDataset,
    VectorDataset,
    VisionClassificationDataset,
    VisionDataset,
)
from .gid15 import GID15
from .globbiomass import GlobBiomass
from .idtrees import IDTReeS
from .inaturalist import INaturalist
from .inria import InriaAerialImageLabeling
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
from .millionaid import MillionAID
from .naip import NAIP
from .nasa_marine_debris import NASAMarineDebris
from .nwpu import VHR10
from .openbuildings import OpenBuildings
from .oscd import OSCD
from .patternnet import PatternNet
from .potsdam import Potsdam2D
from .reforestree import ReforesTree
from .resisc45 import RESISC45
from .seco import SeasonalContrastS2
from .sen12ms import SEN12MS
from .sentinel import Sentinel, Sentinel2
from .so2sat import So2Sat
from .spacenet import (
    SpaceNet,
    SpaceNet1,
    SpaceNet2,
    SpaceNet3,
    SpaceNet4,
    SpaceNet5,
    SpaceNet7,
)
from .ucmerced import UCMerced
from .usavars import USAVars
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
    "AbovegroundLiveWoodyBiomassDensity",
    "AsterGDEM",
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
    "CMSGlobalMangroveCanopy",
    "EDDMapS",
    "Esri2020",
    "EUDEM",
    "GBIF",
    "GlobBiomass",
    "INaturalist",
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
    "OpenBuildings",
    "Sentinel",
    "Sentinel2",
    # NonGeoDataset
    "ADVANCE",
    "BeninSmallHolderCashews",
    "BigEarthNet",
    "CloudCoverDetection",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "CV4AKenyaCropType",
    "DeepGlobeLandCover",
    "DFC2022",
    "EnviroAtlas",
    "ETCI2021",
    "EuroSAT",
    "FAIR1M",
    "ForestDamage",
    "GID15",
    "IDTReeS",
    "InriaAerialImageLabeling",
    "LandCoverAI",
    "LEVIRCDPlus",
    "LoveDA",
    "MillionAID",
    "NASAMarineDebris",
    "OSCD",
    "PatternNet",
    "Potsdam2D",
    "RESISC45",
    "ReforesTree",
    "SeasonalContrastS2",
    "SEN12MS",
    "So2Sat",
    "SpaceNet",
    "SpaceNet1",
    "SpaceNet2",
    "SpaceNet3",
    "SpaceNet4",
    "SpaceNet5",
    "SpaceNet7",
    "TropicalCycloneWindEstimation",
    "UCMerced",
    "USAVars",
    "Vaihingen2D",
    "VHR10",
    "XView2",
    "ZueriCrop",
    # Base classes
    "GeoDataset",
    "IntersectionDataset",
    "NonGeoClassificationDataset",
    "NonGeoDataset",
    "RasterDataset",
    "UnionDataset",
    "VectorDataset",
    "VisionClassificationDataset",
    "VisionDataset",
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
