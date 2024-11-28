# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datamodules."""

from .agrifieldnet import AgriFieldNetDataModule
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
from .flair2 import FLAIR2DataModule
from .ftw import FieldsOfTheWorldDataModule
from .geo import BaseDataModule, GeoDataModule, NonGeoDataModule
from .geonrw import GeoNRWDataModule
from .gid15 import GID15DataModule
from .inria import InriaAerialImageLabelingDataModule
from .iobench import IOBenchDataModule
from .l7irish import L7IrishDataModule
from .l8biome import L8BiomeDataModule
from .landcoverai import LandCoverAI100DataModule, LandCoverAIDataModule
from .levircd import LEVIRCDDataModule, LEVIRCDPlusDataModule
from .loveda import LoveDADataModule
from .naip import NAIPChesapeakeDataModule
from .nasa_marine_debris import NASAMarineDebrisDataModule
from .oscd import OSCDDataModule
from .potsdam import Potsdam2DDataModule
from .quakeset import QuakeSetDataModule
from .resisc45 import RESISC45DataModule
from .seco import SeasonalContrastS2DataModule
from .sen12ms import SEN12MSDataModule
from .sentinel2_cdl import Sentinel2CDLDataModule
from .sentinel2_eurocrops import Sentinel2EuroCropsDataModule
from .sentinel2_nccm import Sentinel2NCCMDataModule
from .sentinel2_south_america_soybean import Sentinel2SouthAmericaSoybeanDataModule
from .skippd import SKIPPDDataModule
from .so2sat import So2SatDataModule
from .southafricacroptype import SouthAfricaCropTypeDataModule
from .spacenet import SpaceNet1DataModule, SpaceNet6DataModule, SpaceNetBaseDataModule
from .ssl4eo import SSL4EOLDataModule, SSL4EOS12DataModule
from .ssl4eo_benchmark import SSL4EOLBenchmarkDataModule
from .sustainbench_crop_yield import SustainBenchCropYieldDataModule
from .ucmerced import UCMercedDataModule
from .usavars import USAVarsDataModule
from .utils import MisconfigurationException
from .vaihingen import Vaihingen2DDataModule
from .vhr10 import VHR10DataModule
from .xview import XView2DataModule

__all__ = (
    'AgriFieldNetDataModule',
    'BaseDataModule',
    'BigEarthNetDataModule',
    'CaBuArDataModule',
    'CaFFeDataModule',
    'ChaBuDDataModule',
    'ChesapeakeCVPRDataModule',
    'COWCCountingDataModule',
    'DeepGlobeLandCoverDataModule',
    'DigitalTyphoonDataModule',
    'ETCI2021DataModule',
    'EuroSAT100DataModule',
    'EuroSATDataModule',
    'EuroSATSpatialDataModule',
    'FAIR1MDataModule',
    'FieldsOfTheWorldDataModule',
    'FireRiskDataModule',
    'FLAIR2DataModule',
    'GeoDataModule',
    'GeoNRWDataModule',
    'GID15DataModule',
    'InriaAerialImageLabelingDataModule',
    'IOBenchDataModule',
    'L7IrishDataModule',
    'L8BiomeDataModule',
    'LandCoverAI100DataModule',
    'LandCoverAIDataModule',
    'LEVIRCDDataModule',
    'LEVIRCDPlusDataModule',
    'LoveDADataModule',
    'MisconfigurationException',
    'NAIPChesapeakeDataModule',
    'NASAMarineDebrisDataModule',
    'NonGeoDataModule',
    'OSCDDataModule',
    'Potsdam2DDataModule',
    'QuakeSetDataModule',
    'RESISC45DataModule',
    'SeasonalContrastS2DataModule',
    'SEN12MSDataModule',
    'Sentinel2CDLDataModule',
    'Sentinel2EuroCropsDataModule',
    'Sentinel2NCCMDataModule',
    'Sentinel2SouthAmericaSoybeanDataModule',
    'SKIPPDDataModule',
    'So2SatDataModule',
    'SouthAfricaCropTypeDataModule',
    'SpaceNet1DataModule',
    'SpaceNet6DataModule',
    'SpaceNetBaseDataModule',
    'SSL4EOLBenchmarkDataModule',
    'SSL4EOLDataModule',
    'SSL4EOS12DataModule',
    'SustainBenchCropYieldDataModule',
    'TropicalCycloneDataModule',
    'UCMercedDataModule',
    'USAVarsDataModule',
    'Vaihingen2DDataModule',
    'VHR10DataModule',
    'XView2DataModule',
)
