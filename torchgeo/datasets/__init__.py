from .benin_cashews import BeninSmallHolderCashews
from .cowc import COWC, COWCCounting, COWCDetection
from .cv4a_kenya_crop_type import CV4AKenyaCropType
from .cyclone import TropicalCycloneWindEstimation
from .geo import GeoDataset, VisionDataset, ZipDataset
from .landcoverai import LandCoverAI
from .nwpu import VHR10
from .sen12ms import SEN12MS
from .so2sat import So2Sat

# https://stackoverflow.com/questions/40018681
BeninSmallHolderCashews.__module__ = "torchgeo.datasets"
COWC.__module__ = "torchgeo.datasets"
COWCCounting.__module__ = "torchgeo.datasets"
COWCDetection.__module__ = "torchgeo.datasets"
CV4AKenyaCropType.__module__ = "torchgeo.datasets"
TropicalCycloneWindEstimation.__module__ = "torchgeo.datasets"
GeoDataset.__module__ = "torchgeo.datasets"
VisionDataset.__module__ = "torchgeo.datasets"
ZipDataset.__module__ = "torchgeo.datasets"
LandCoverAI.__module__ = "torchgeo.datasets"
VHR10.__module__ = "torchgeo.datasets"
SEN12MS.__module__ = "torchgeo.datasets"
So2Sat.__module__ = "torchgeo.datasets"

__all__ = (
    "BeninSmallHolderCashews",
    "COWC",
    "COWCCounting",
    "COWCDetection",
    "CV4AKenyaCropType",
    "GeoDataset",
    "LandCoverAI",
    "SEN12MS",
    "So2Sat",
    "TropicalCycloneWindEstimation",
    "VHR10",
    "VisionDataset",
    "ZipDataset",
)
