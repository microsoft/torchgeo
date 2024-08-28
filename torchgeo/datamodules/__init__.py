# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datamodules."""

from .agrifieldnet import AgriFieldNetDataModule
from .bigearthnet import BigEarthNetDataModule
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
from .geo import BaseDataModule, GeoDataModule, NonGeoDataModule
from .gid15 import GID15DataModule
from .inria import InriaAerialImageLabelingDataModule
from .iobench import IOBenchDataModule
from .l7irish import L7IrishDataModule
from .l8biome import L8BiomeDataModule
from .landcoverai import LandCoverAIDataModule
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
from .spacenet import SpaceNet1DataModule
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
    # GeoDataset
    'AgriFieldNetDataModule',
    'ChesapeakeCVPRDataModule',
    'IOBenchDataModule',
    'L7IrishDataModule',
    'L8BiomeDataModule',
    'NAIPChesapeakeDataModule',
    'Sentinel2CDLDataModule',
    'Sentinel2EuroCropsDataModule',
    'Sentinel2NCCMDataModule',
    'Sentinel2SouthAmericaSoybeanDataModule',
    'SouthAfricaCropTypeDataModule',
    # NonGeoDataset
    'BigEarthNetDataModule',
    'ChaBuDDataModule',
    'COWCCountingDataModule',
    'DeepGlobeLandCoverDataModule',
    'DigitalTyphoonDataModule',
    'ETCI2021DataModule',
    'EuroSATDataModule',
    'EuroSATSpatialDataModule',
    'EuroSAT100DataModule',
    'FAIR1MDataModule',
    'FireRiskDataModule',
    'GID15DataModule',
    'InriaAerialImageLabelingDataModule',
    'LandCoverAIDataModule',
    'LEVIRCDDataModule',
    'LEVIRCDPlusDataModule',
    'LoveDADataModule',
    'NASAMarineDebrisDataModule',
    'OSCDDataModule',
    'Potsdam2DDataModule',
    'QuakeSetDataModule',
    'RESISC45DataModule',
    'SeasonalContrastS2DataModule',
    'SEN12MSDataModule',
    'SKIPPDDataModule',
    'So2SatDataModule',
    'SpaceNet1DataModule',
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
    # Base classes
    'BaseDataModule',
    'GeoDataModule',
    'NonGeoDataModule',
    # Utilities
    'MisconfigurationException',
)
