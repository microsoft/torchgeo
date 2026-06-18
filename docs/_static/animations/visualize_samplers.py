# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Visualize samplers."""

import itertools
from pathlib import Path

from geopandas import GeoDataFrame
from pandas import IntervalIndex, Timedelta, Timestamp
from shapely import Polygon

from torchgeo.datasets import RasterDataset
from torchgeo.samplers import (
    GriddedPatchSampler,
    RandomPatchSampler,
    RandomPeriodSampler,
    RandomTimedeltaSampler,
    RandomTimestampSampler,
    SequentialPeriodSampler,
    SequentialTimedeltaSampler,
    SequentialTimestampSampler,
)


class ToyDataset(RasterDataset):
    """Toy dataset for sampler visualization."""

    def __init__(self) -> None:
        """Initialize a new ToyDataset instance."""
        datetimes = [
            (Timestamp(2026, 7, 1), Timestamp(2026, 7, 8)),
            (Timestamp(2026, 7, 16), Timestamp(2026, 7, 23)),
            (Timestamp(2026, 8, 1), Timestamp(2026, 8, 8)),
            (Timestamp(2026, 8, 16), Timestamp(2026, 8, 23)),
        ]
        geometries = [
            Polygon([(10, 0), (20, 10), (10, 20), (0, 10), (10, 0)]),
            Polygon([(20, 0), (30, 10), (20, 20), (10, 10), (20, 0)]),
            Polygon([(10, 0), (20, 10), (10, 20), (0, 10), (10, 0)]),
            Polygon([(20, 0), (30, 10), (20, 20), (10, 10), (20, 0)]),
        ]
        index = IntervalIndex.from_tuples(datetimes, closed='both', name='datetime')
        self.index = GeoDataFrame(index=index, geometry=geometries)
        self._res = (1, 1)


dataset = ToyDataset()

spatial_samplers = [
    RandomPatchSampler(dataset, size=3, generator=0),
    GriddedPatchSampler(dataset, size=3, stride=2),
]
temporal_samplers = [
    RandomTimestampSampler(dataset, generator=0),
    SequentialTimestampSampler(dataset),
    RandomTimedeltaSampler(dataset, delta=Timedelta('31D'), generator=0),
    SequentialTimedeltaSampler(dataset, delta=Timedelta('31D')),
    RandomPeriodSampler(dataset, freq='M', generator=0),
    SequentialPeriodSampler(dataset, freq='M'),
]

directory = Path(__file__).resolve().parent
for sampler in spatial_samplers + temporal_samplers:
    filename = f'{sampler.__class__.__name__}.gif'
    print(filename)
    ani = sampler.plot()
    ani.save(directory / filename)

for spatial, temporal in itertools.product(spatial_samplers, temporal_samplers):
    filename = f'{spatial.__class__.__name__}_{temporal.__class__.__name__}.gif'
    print(filename)
    sampler = spatial @ temporal
    ani = sampler.plot()
    ani.save(directory / filename)
