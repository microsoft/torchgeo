# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datasets."""

from .advance import ADVANCE
from .agb_live_woody_density import AbovegroundLiveWoodyBiomassDensity
from .agrifieldnet import AgriFieldNet
from .airphen import Airphen
from .astergdem import AsterGDEM
from .benin_cashews import BeninSmallHolderCashews
from .bigearthnet import BigEarthNet
from .biomassters import BioMassters
from .cbf import CanadianBuildingFootprints
from .cdl import CDL
from .chabud import ChaBuD
from .chesapeake import (
    Chesapeake,
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
from .cropharvest import CropHarvest
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import TropicalCyclone
from .deepglobelandcover import DeepGlobeLandCover
from .dfc2022 import DFC2022
from .eddmaps import EDDMapS
from .enviroatlas import EnviroAtlas
from .errors import DatasetNotFoundError, DependencyNotFoundError, RGBBandsMissingError
from .esri2020 import Esri2020
from .etci2021 import ETCI2021
from .eudem import EUDEM
from .eurocrops import EuroCrops
from .eurosat import EuroSAT, EuroSAT100, EuroSATSpatial
from .fair1m import FAIR1M
from .fire_risk import FireRisk
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
)
from .gid15 import GID15
from .globbiomass import GlobBiomass
from .idtrees import IDTReeS
from .inaturalist import INaturalist
from .inria import InriaAerialImageLabeling
from .iobench import IOBench
from .l7irish import L7Irish
from .l8biome import L8Biome
from .landcoverai import LandCoverAI, LandCoverAIBase, LandCoverAIGeo
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
from .levircd import LEVIRCD, LEVIRCDBase, LEVIRCDPlus
from .loveda import LoveDA
from .mapinwild import MapInWild
from .millionaid import MillionAID
from .naip import NAIP
from .nasa_marine_debris import NASAMarineDebris
from .nccm import NCCM
from .nlcd import NLCD
from .openbuildings import OpenBuildings
from .oscd import OSCD
from .pastis import PASTIS
from .patternnet import PatternNet
from .potsdam import Potsdam2D
from .prisma import PRISMA
from .quakeset import QuakeSet
from .reforestree import ReforesTree
from .resisc45 import RESISC45
from .rwanda_field_boundary import RwandaFieldBoundary
from .seasonet import SeasoNet
from .seco import SeasonalContrastS2
from .sen12ms import SEN12MS
from .sentinel import Sentinel, Sentinel1, Sentinel2
from .skippd import SKIPPD
from .skyscript import SkyScript
from .so2sat import So2Sat
from .south_africa_crop_type import SouthAfricaCropType
from .south_america_soybean import SouthAmericaSoybean
from .spacenet import (
    SpaceNet,
    SpaceNet1,
    SpaceNet2,
    SpaceNet3,
    SpaceNet4,
    SpaceNet5,
    SpaceNet6,
    SpaceNet7,
    SpaceNet8,
)
from .splits import (
    random_bbox_assignment,
    random_bbox_splitting,
    random_grid_cell_assignment,
    roi_split,
    time_series_split,
)
from .ssl4eo import SSL4EO, SSL4EOL, SSL4EOS12
from .ssl4eo_benchmark import SSL4EOLBenchmark
from .sustainbench_crop_yield import SustainBenchCropYield
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
from .vhr10 import VHR10
from .western_usa_live_fuel_moisture import WesternUSALiveFuelMoisture
from .xview import XView2
from .zuericrop import ZueriCrop

__all__ = (
    # GeoDataset
    'AbovegroundLiveWoodyBiomassDensity',
    'AgriFieldNet',
    'Airphen',
    'AsterGDEM',
    'CanadianBuildingFootprints',
    'CDL',
    'Chesapeake',
    'Chesapeake7',
    'Chesapeake13',
    'ChesapeakeDC',
    'ChesapeakeDE',
    'ChesapeakeMD',
    'ChesapeakeNY',
    'ChesapeakePA',
    'ChesapeakeVA',
    'ChesapeakeWV',
    'ChesapeakeCVPR',
    'CMSGlobalMangroveCanopy',
    'CropHarvest',
    'EDDMapS',
    'Esri2020',
    'EuroCrops',
    'EUDEM',
    'GBIF',
    'GlobBiomass',
    'INaturalist',
    'IOBench',
    'L7Irish',
    'L8Biome',
    'LandCoverAIBase',
    'LandCoverAIGeo',
    'Landsat',
    'Landsat1',
    'Landsat2',
    'Landsat3',
    'Landsat4MSS',
    'Landsat4TM',
    'Landsat5MSS',
    'Landsat5TM',
    'Landsat7',
    'Landsat8',
    'Landsat9',
    'NAIP',
    'NCCM',
    'NLCD',
    'OpenBuildings',
    'PRISMA',
    'Sentinel',
    'Sentinel1',
    'Sentinel2',
    'SouthAfricaCropType',
    'SouthAmericaSoybean',
    # NonGeoDataset
    'ADVANCE',
    'BeninSmallHolderCashews',
    'BigEarthNet',
    'BioMassters',
    'ChaBuD',
    'CloudCoverDetection',
    'COWC',
    'COWCCounting',
    'COWCDetection',
    'CV4AKenyaCropType',
    'DeepGlobeLandCover',
    'DFC2022',
    'EnviroAtlas',
    'ETCI2021',
    'EuroSAT',
    'EuroSATSpatial',
    'EuroSAT100',
    'FAIR1M',
    'FireRisk',
    'ForestDamage',
    'GID15',
    'IDTReeS',
    'InriaAerialImageLabeling',
    'LandCoverAI',
    'LEVIRCD',
    'LEVIRCDBase',
    'LEVIRCDPlus',
    'LoveDA',
    'MapInWild',
    'MillionAID',
    'NASAMarineDebris',
    'OSCD',
    'PASTIS',
    'PatternNet',
    'Potsdam2D',
    'QuakeSet',
    'RESISC45',
    'ReforesTree',
    'RwandaFieldBoundary',
    'SeasonalContrastS2',
    'SeasoNet',
    'SEN12MS',
    'SKIPPD',
    'SkyScript',
    'So2Sat',
    'SpaceNet',
    'SpaceNet1',
    'SpaceNet2',
    'SpaceNet3',
    'SpaceNet4',
    'SpaceNet5',
    'SpaceNet6',
    'SpaceNet7',
    'SpaceNet8',
    'SSL4EO',
    'SSL4EOLBenchmark',
    'SSL4EOL',
    'SSL4EOS12',
    'SustainBenchCropYield',
    'TropicalCyclone',
    'UCMerced',
    'USAVars',
    'Vaihingen2D',
    'VHR10',
    'WesternUSALiveFuelMoisture',
    'XView2',
    'ZueriCrop',
    # Base classes
    'GeoDataset',
    'IntersectionDataset',
    'NonGeoClassificationDataset',
    'NonGeoDataset',
    'RasterDataset',
    'UnionDataset',
    'VectorDataset',
    # Utilities
    'BoundingBox',
    'concat_samples',
    'merge_samples',
    'stack_samples',
    'unbind_samples',
    # Splits
    'random_bbox_assignment',
    'random_bbox_splitting',
    'random_grid_cell_assignment',
    'roi_split',
    'time_series_split',
    # Errors
    'DatasetNotFoundError',
    'DependencyNotFoundError',
    'RGBBandsMissingError',
)
