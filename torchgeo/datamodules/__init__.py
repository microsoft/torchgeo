# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datamodules."""

from .agrifieldnet import AgriFieldNetDataModule
from .air_quality import AirQualityDataModule
from .bigearthnet import BigEarthNetDataModule
from .cabuar import CaBuArDataModule
from .caffe import CaFFeDataModule
from .chabud import ChaBuDDataModule
from .chesapeake import ChesapeakeCVPRDataModule
from .cowc import COWCCountingDataModule
from .cyclone import TropicalCycloneDataModule
from .deepglobelandcover import DeepGlobeLandCoverDataModule
from .digital_typhoon import DigitalTyphoonDataModule
from .etci2021 import ETCI2021DataModule
from .eurosat import EuroSAT100DataModule, EuroSATDataModule, EuroSATSpatialDataModule
from .fair1m import FAIR1MDataModule
from .fire_risk import FireRiskDataModule
from .ftw import FieldsOfTheWorldDataModule
from .geo import BaseDataModule, GeoDataModule, NonGeoDataModule
from .geonrw import GeoNRWDataModule
from .gid15 import GID15DataModule
from .hyspecnet import HySpecNet11kDataModule
from .inria import InriaAerialImageLabelingDataModule
from .iobench import IOBenchDataModule
from .l7irish import L7IrishDataModule
from .l8biome import L8BiomeDataModule
from .landcoverai import LandCoverAI100DataModule, LandCoverAIDataModule
from .levircd import LEVIRCDDataModule, LEVIRCDPlusDataModule
from .loveda import LoveDADataModule
from .mmflood import MMFloodDataModule
from .naip import NAIPChesapeakeDataModule
from .nasa_marine_debris import NASAMarineDebrisDataModule
from .oscd import OSCDDataModule
from .patternnet import PatternNetDataModule
from .potsdam import Potsdam2DDataModule
from .quakeset import QuakeSetDataModule
from .reforestree import ReforesTreeDataModule
from .resisc45 import RESISC45DataModule
from .seco import SeasonalContrastS2DataModule
from .sen12ms import SEN12MSDataModule
from .sentinel2_cdl import Sentinel2CDLDataModule
from .sentinel2_eurocrops import Sentinel2EuroCropsDataModule
from .sentinel2_nccm import Sentinel2NCCMDataModule
from .sentinel2_south_america_soybean import Sentinel2SouthAmericaSoybeanDataModule
from .skippd import SKIPPDDataModule
from .so2sat import So2SatDataModule
from .solar_plants_brazil import SolarPlantsBrazilDataModule
from .southafricacroptype import SouthAfricaCropTypeDataModule
from .spacenet import SpaceNet1DataModule, SpaceNet6DataModule, SpaceNetBaseDataModule
from .ssl4eo import SSL4EOLDataModule, SSL4EOS12DataModule
from .ssl4eo_benchmark import SSL4EOLBenchmarkDataModule
from .substation import SubstationDataModule
from .sustainbench_crop_yield import SustainBenchCropYieldDataModule
from .treesatai import TreeSatAIDataModule
from .ucmerced import UCMercedDataModule
from .usavars import USAVarsDataModule
from .utils import MisconfigurationException
from .vaihingen import Vaihingen2DDataModule
from .vhr10 import VHR10DataModule
from .xview import XView2DataModule

__all__ = (
    'AgriFieldNetDataModule',
    'AirQualityDataModule',
    'BaseDataModule',
    'BigEarthNetDataModule',
    'COWCCountingDataModule',
    'CaBuArDataModule',
    'CaFFeDataModule',
    'ChaBuDDataModule',
    'ChesapeakeCVPRDataModule',
    'DeepGlobeLandCoverDataModule',
    'DigitalTyphoonDataModule',
    'ETCI2021DataModule',
    'EuroSAT100DataModule',
    'EuroSATDataModule',
    'EuroSATSpatialDataModule',
    'FAIR1MDataModule',
    'FieldsOfTheWorldDataModule',
    'FireRiskDataModule',
    'GID15DataModule',
    'GeoDataModule',
    'GeoNRWDataModule',
    'HySpecNet11kDataModule',
    'IOBenchDataModule',
    'InriaAerialImageLabelingDataModule',
    'L7IrishDataModule',
    'L8BiomeDataModule',
    'LEVIRCDDataModule',
    'LEVIRCDPlusDataModule',
    'LandCoverAI100DataModule',
    'LandCoverAIDataModule',
    'LoveDADataModule',
    'MMFloodDataModule',
    'MisconfigurationException',
    'NAIPChesapeakeDataModule',
    'NASAMarineDebrisDataModule',
    'NonGeoDataModule',
    'OSCDDataModule',
    'PatternNetDataModule',
    'Potsdam2DDataModule',
    'QuakeSetDataModule',
    'RESISC45DataModule',
    'ReforesTreeDataModule',
    'SEN12MSDataModule',
    'SKIPPDDataModule',
    'SSL4EOLBenchmarkDataModule',
    'SSL4EOLDataModule',
    'SSL4EOS12DataModule',
    'SeasonalContrastS2DataModule',
    'Sentinel2CDLDataModule',
    'Sentinel2EuroCropsDataModule',
    'Sentinel2NCCMDataModule',
    'Sentinel2SouthAmericaSoybeanDataModule',
    'So2SatDataModule',
    'SolarPlantsBrazilDataModule',
    'SouthAfricaCropTypeDataModule',
    'SpaceNet1DataModule',
    'SpaceNet6DataModule',
    'SpaceNetBaseDataModule',
    'SubstationDataModule',
    'SustainBenchCropYieldDataModule',
    'TreeSatAIDataModule',
    'TropicalCycloneDataModule',
    'UCMercedDataModule',
    'USAVarsDataModule',
    'VHR10DataModule',
    'Vaihingen2DDataModule',
    'XView2DataModule',
)
