"""TorchGeo trainers."""

from .cyclone import CycloneDataModule, CycloneSimpleRegressionTask
from .sen12ms import SEN12MSDataModule, SEN12MSSegmentationTask

__all__ = (
    "CycloneDataModule",
    "CycloneSimpleRegressionTask",
    "SEN12MSDataModule",
    "SEN12MSSegmentationTask",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
