"""TorchGeo specific models."""

from .fcn import FCN

__all__ = ("FCN",)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.models"
