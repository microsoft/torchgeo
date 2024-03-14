# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo datamodules."""

from .bigearthnet import BigEarthNetDataModule
from .chabud import ChaBuDDataModule
from .chesapeake import ChesapeakeCVPRDataModule
from .cowc import COWCCountingDataModule
from .cyclone import TropicalCycloneDataModule
from .deepglobelandcover import DeepGlobeLandCoverDataModule
from .etci2021 import ETCI2021DataModule
from .eurosat import EuroSAT100DataModule, EuroSATDataModule
from .fair1m import FAIR1MDataModule
from .fire_risk import FireRiskDataModule
from .geo import BaseDataModule, GeoDataModule, NonGeoDataModule
from .gid15 import GID15DataModule
from .inria import InriaAerialImageLabelingDataModule
from .l7irish import L7IrishDataModule
from .l8biome import L8BiomeDataModule
from .landcoverai import LandCoverAIDataModule
from .levircd import LEVIRCDDataModule, LEVIRCDPlusDataModule
from .loveda import LoveDADataModule
from .naip import NAIPChesapeakeDataModule
from .nasa_marine_debris import NASAMarineDebrisDataModule
from .oscd import OSCDDataModule
from .potsdam import Potsdam2DDataModule
from .resisc45 import RESISC45DataModule
from .seco import SeasonalContrastS2DataModule
from .sen12ms import SEN12MSDataModule
from .sentinel2cdl import Sentinel2CDLDataModule
from .skippd import SKIPPDDataModule
from .so2sat import So2SatDataModule
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
    "Sentinel2CDLDataModule",
    "ChesapeakeCVPRDataModule",
    "L7IrishDataModule",
    "L8BiomeDataModule",
    "NAIPChesapeakeDataModule",
    # NonGeoDataset
    "BigEarthNetDataModule",
    "ChaBuDDataModule",
    "COWCCountingDataModule",
    "DeepGlobeLandCoverDataModule",
    "ETCI2021DataModule",
    "EuroSATDataModule",
    "EuroSAT100DataModule",
    "FAIR1MDataModule",
    "FireRiskDataModule",
    "GID15DataModule",
    "InriaAerialImageLabelingDataModule",
    "LandCoverAIDataModule",
    "LEVIRCDDataModule",
    "LEVIRCDPlusDataModule",
    "LoveDADataModule",
    "NASAMarineDebrisDataModule",
    "OSCDDataModule",
    "Potsdam2DDataModule",
    "RESISC45DataModule",
    "SeasonalContrastS2DataModule",
    "SEN12MSDataModule",
    "SKIPPDDataModule",
    "So2SatDataModule",
    "SpaceNet1DataModule",
    "SSL4EOLBenchmarkDataModule",
    "SSL4EOLDataModule",
    "SSL4EOS12DataModule",
    "SustainBenchCropYieldDataModule",
    "TropicalCycloneDataModule",
    "UCMercedDataModule",
    "USAVarsDataModule",
    "Vaihingen2DDataModule",
    "VHR10DataModule",
    "XView2DataModule",
    # Base classes
    "BaseDataModule",
    "GeoDataModule",
    "NonGeoDataModule",
    # Utilities
    "MisconfigurationException",
)
