from .benin_cashews import BeninSmallHolderCashews
from .cdl import CDL
from .cowc import COWC, COWCCounting, COWCDetection
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import TropicalCycloneWindEstimation
from .geo import GeoDataset, VisionDataset, ZipDataset
from .landcoverai import LandCoverAI
from .landsat import (
    Landsat,
    Landsat1_3,
    Landsat4_5MSS,
    Landsat4_5TM,
    Landsat7,
    Landsat8_9,
)
from .nwpu import VHR10
from .sen12ms import SEN12MS
from .sentinel import Sentinel, Sentinel2
from .so2sat import So2Sat
from .utils import BoundingBox, collate_dict

__all__ = (
    "BeninSmallHolderCashews",
    "BoundingBox",
    "CDL",
    "collate_dict",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "CV4AKenyaCropType",
    "GeoDataset",
    "LandCoverAI",
    "Landsat",
    "Landsat1_3",
    "Landsat4_5MSS",
    "Landsat4_5TM",
    "Landsat7",
    "Landsat8_9",
    "SEN12MS",
    "Sentinel",
    "Sentinel2",
    "So2Sat",
    "TropicalCycloneWindEstimation",
    "VHR10",
    "VisionDataset",
    "ZipDataset",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.datasets"
