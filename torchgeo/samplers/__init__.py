# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""TorchGeo samplers."""

from .batch import BatchGeoSampler, RandomBatchGeoSampler
from .single import GeoSampler, GridGeoSampler, RandomGeoSampler

__all__ = (
    # Samplers
    "GridGeoSampler",
    "RandomGeoSampler",
    # Batch samplers
    "RandomBatchGeoSampler",
    # Base classes
    "GeoSampler",
    "BatchGeoSampler",
)

# https://stackoverflow.com/questions/40018681
for module in __all__:
    globals()[module].__module__ = "torchgeo.samplers"
