# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datasets."""

from .advance import ADVANCE
from .agb_live_woody_density import AbovegroundLiveWoodyBiomassDensity
from .agrifieldnet import AgriFieldNet
from .airphen import Airphen
from .astergdem import AsterGDEM
from .benin_cashews import BeninSmallHolderCashews
from .bigearthnet import BigEarthNet, BigEarthNetV2
from .biomassters import BioMassters
from .cabuar import CaBuAr
from .caffe import CaFFe
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
from .digital_typhoon import DigitalTyphoon
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
from .ftw import FieldsOfTheWorld
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
from .geonrw import GeoNRW
from .gid15 import GID15
from .globbiomass import GlobBiomass
from .hyspecnet import HySpecNet11k
from .idtrees import IDTReeS
from .inaturalist import INaturalist
from .inria import InriaAerialImageLabeling
from .iobench import IOBench
from .l7irish import L7Irish
from .l8biome import L8Biome
from .landcoverai import LandCoverAI, LandCoverAI100, LandCoverAIBase, LandCoverAIGeo
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
from .mdas import MDAS
from .millionaid import MillionAID
from .mmearth import MMEarth
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
from .satlas import SatlasPretrain
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
from .treesatai import TreeSatAI
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
    'ADVANCE',
    'CDL',
    'COWC',
    'DFC2022',
    'ETCI2021',
    'EUDEM',
    'FAIR1M',
    'GBIF',
    'GID15',
    'LEVIRCD',
    'MDAS',
    'NAIP',
    'NCCM',
    'NLCD',
    'OSCD',
    'PASTIS',
    'PRISMA',
    'RESISC45',
    'SEN12MS',
    'SKIPPD',
    'SSL4EO',
    'SSL4EOL',
    'SSL4EOS12',
    'VHR10',
    'AbovegroundLiveWoodyBiomassDensity',
    'AgriFieldNet',
    'Airphen',
    'AsterGDEM',
    'BeninSmallHolderCashews',
    'BigEarthNet',
    'BigEarthNetV2',
    'BioMassters',
    'BoundingBox',
    'CMSGlobalMangroveCanopy',
    'COWCCounting',
    'COWCDetection',
    'CV4AKenyaCropType',
    'CaBuAr',
    'CaFFe',
    'CanadianBuildingFootprints',
    'ChaBuD',
    'Chesapeake',
    'ChesapeakeCVPR',
    'ChesapeakeDC',
    'ChesapeakeDE',
    'ChesapeakeMD',
    'ChesapeakeNY',
    'ChesapeakePA',
    'ChesapeakeVA',
    'ChesapeakeWV',
    'CloudCoverDetection',
    'CropHarvest',
    'DatasetNotFoundError',
    'DeepGlobeLandCover',
    'DependencyNotFoundError',
    'DigitalTyphoon',
    'EDDMapS',
    'EnviroAtlas',
    'Esri2020',
    'EuroCrops',
    'EuroSAT',
    'EuroSAT100',
    'EuroSATSpatial',
    'FieldsOfTheWorld',
    'FireRisk',
    'ForestDamage',
    'GeoDataset',
    'GeoNRW',
    'GlobBiomass',
    'HySpecNet11k',
    'IDTReeS',
    'INaturalist',
    'IOBench',
    'InriaAerialImageLabeling',
    'IntersectionDataset',
    'L7Irish',
    'L8Biome',
    'LEVIRCDBase',
    'LEVIRCDPlus',
    'LandCoverAI',
    'LandCoverAI100',
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
    'LoveDA',
    'MMEarth',
    'MapInWild',
    'MillionAID',
    'NASAMarineDebris',
    'NonGeoClassificationDataset',
    'NonGeoDataset',
    'OpenBuildings',
    'PatternNet',
    'Potsdam2D',
    'QuakeSet',
    'RGBBandsMissingError',
    'RasterDataset',
    'ReforesTree',
    'RwandaFieldBoundary',
    'SSL4EOLBenchmark',
    'SatlasPretrain',
    'SeasoNet',
    'SeasonalContrastS2',
    'Sentinel',
    'Sentinel1',
    'Sentinel2',
    'SkyScript',
    'So2Sat',
    'SouthAfricaCropType',
    'SouthAmericaSoybean',
    'SpaceNet',
    'SpaceNet1',
    'SpaceNet2',
    'SpaceNet3',
    'SpaceNet4',
    'SpaceNet5',
    'SpaceNet6',
    'SpaceNet7',
    'SpaceNet8',
    'SustainBenchCropYield',
    'TreeSatAI',
    'TropicalCyclone',
    'UCMerced',
    'USAVars',
    'UnionDataset',
    'Vaihingen2D',
    'VectorDataset',
    'WesternUSALiveFuelMoisture',
    'XView2',
    'ZueriCrop',
    'concat_samples',
    'merge_samples',
    'random_bbox_assignment',
    'random_bbox_splitting',
    'random_grid_cell_assignment',
    'roi_split',
    'stack_samples',
    'time_series_split',
    'unbind_samples',
)
