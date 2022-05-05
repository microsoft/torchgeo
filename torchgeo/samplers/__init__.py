# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

from .batch import BatchGeoSampler, RandomBatchGeoSampler, TripletBatchGeoSampler
from .constants import Units
from .single import GeoSampler, GridGeoSampler, PreChippedGeoSampler, RandomGeoSampler

__all__ = (
    # Samplers
    "GridGeoSampler",
    "PreChippedGeoSampler",
    "RandomGeoSampler",
    # Batch samplers
    "RandomBatchGeoSampler",
    # Base classes
    "GeoSampler",
    "BatchGeoSampler",
    # Constants
    "Units",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.samplers"
