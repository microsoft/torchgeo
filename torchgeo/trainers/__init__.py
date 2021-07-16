"""TorchGeo trainers."""

from .cyclone import CycloneDataModule, CycloneSimpleRegressionTask

__all__ = ("CycloneDataModule", "CycloneSimpleRegressionTask")

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.trainers"
