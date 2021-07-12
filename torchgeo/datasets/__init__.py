from .benin_cashews import BeninSmallHolderCashews
from .cowc import COWC, COWCCounting, COWCDetection
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import TropicalCycloneWindEstimation
from .geo import BoundingBox, GeoDataset, VisionDataset, ZipDataset
from .landcoverai import LandCoverAI
from .nwpu import VHR10
from .sen12ms import SEN12MS
from .sentinel import Sentinel, Sentinel2
from .so2sat import So2Sat

__all__ = (
    "BeninSmallHolderCashews",
    "BoundingBox",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "CV4AKenyaCropType",
    "GeoDataset",
    "LandCoverAI",
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
