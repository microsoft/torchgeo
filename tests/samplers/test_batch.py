# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from collections.abc import Iterator, Sequence
from itertools import product

import pandas as pd
import pytest
import shapely
import torch
from _pytest.fixtures import SubRequest
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import Geometry
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset, stack_samples
from torchgeo.datasets.utils import GeoSlice
from torchgeo.samplers import BatchGeoSampler, RandomBatchGeoSampler, Units

MINT = pd.Timestamp(2025, 4, 24)
MAXT = pd.Timestamp(2025, 4, 25)


class CustomBatchGeoSampler(BatchGeoSampler):
    def __iter__(self) -> Iterator[list[GeoSlice]]:
        for i in range(2):
            yield [(slice(j, j), slice(j, j), slice(MINT, MAXT)) for j in range(2)]


class CustomGeoDataset(GeoDataset):
    def __init__(
        self, geometry: Sequence[Geometry], res: tuple[float, float] = (10, 10)
    ) -> None:
        intervals = [(MINT, MAXT)] * len(geometry)
        index = pd.IntervalIndex.from_tuples(intervals, closed='both', name='datetime')
        crs = CRS.from_epsg(3005)
        self.index = GeoDataFrame(index=index, geometry=geometry, crs=crs)
        self.res = res

    def __getitem__(self, query: GeoSlice) -> dict[str, GeoSlice]:
        return {'index': query}


class TestBatchGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        geometry = [shapely.box(0, 0, 100, 100)]
        return CustomGeoDataset(geometry)

    @pytest.fixture(scope='function')
    def sampler(self, dataset: CustomGeoDataset) -> CustomBatchGeoSampler:
        return CustomBatchGeoSampler(dataset)

    def test_iter(self, sampler: CustomBatchGeoSampler) -> None:
        expected = [
            (slice(0, 0), slice(0, 0), slice(MINT, MAXT)),
            (slice(1, 1), slice(1, 1), slice(MINT, MAXT)),
        ]
        assert next(iter(sampler)) == expected

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self,
        dataset: CustomGeoDataset,
        sampler: CustomBatchGeoSampler,
        num_workers: int,
    ) -> None:
        dl = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=stack_samples,
        )
        for _ in dl:
            continue

    def test_abstract(self, dataset: CustomGeoDataset) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BatchGeoSampler(dataset)  # type: ignore[abstract]


class TestRandomBatchGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        geometry = [shapely.box(0, 0, 100, 100), shapely.box(0, 0, 100, 100)]
        return CustomGeoDataset(geometry)

    @pytest.fixture(
        scope='function',
        params=product([3, 4.5, (2, 2), (3, 4.5), (4.5, 3)], [Units.PIXELS, Units.CRS]),
    )
    def sampler(
        self, dataset: CustomGeoDataset, request: SubRequest
    ) -> RandomBatchGeoSampler:
        size, units = request.param
        return RandomBatchGeoSampler(
            dataset, size, batch_size=2, length=10, units=units
        )

    def test_iter(self, sampler: RandomBatchGeoSampler) -> None:
        for batch in sampler:
            for x, y, t in batch:
                bbox = shapely.box(x.start, y.start, x.stop, y.stop)
                assert sampler.roi.contains(bbox)

                assert math.isclose(x.stop - x.start, sampler.size[1])
                assert math.isclose(y.stop - y.start, sampler.size[0])

    def test_len(self, sampler: RandomBatchGeoSampler) -> None:
        assert len(sampler) == sampler.length // sampler.batch_size

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = shapely.box(0, 0, 50, 50)
        sampler = RandomBatchGeoSampler(dataset, 2, 2, 10, roi=roi)
        for batch in sampler:
            for x, y, t in batch:
                bbox = shapely.box(x.start, y.start, x.stop, y.stop)
                assert roi.contains(bbox)

    def test_toi(self, dataset: CustomGeoDataset) -> None:
        toi = pd.Interval(pd.Timestamp(2025, 4, 24, 3), pd.Timestamp(2025, 4, 24, 9))
        sampler = RandomBatchGeoSampler(dataset, 2, 2, 10, toi=toi)
        for batch in sampler:
            for x, y, t in batch:
                bbox = pd.Interval(t.start, t.stop)
                assert toi.overlaps(bbox)

    def test_small_area(self) -> None:
        geometry = [shapely.box(0, 0, 10, 10), shapely.box(20, 20, 21, 21)]
        ds = CustomGeoDataset(geometry, res=(1, 1))
        sampler = RandomBatchGeoSampler(ds, 2, 2, 10)
        for _ in sampler:
            continue

    def test_point_data(self) -> None:
        geometry = [shapely.Point(0, 0), shapely.Point(1, 1)]
        ds = CustomGeoDataset(geometry)
        sampler = RandomBatchGeoSampler(ds, 0, 2, 10)
        for _ in sampler:
            continue

    def test_weighted_sampling(self) -> None:
        geometry = [shapely.Point(0, 0), shapely.box(0, 0, 10, 10)]
        ds = CustomGeoDataset(geometry)
        sampler = RandomBatchGeoSampler(ds, 1, 2, 10)
        for batch in sampler:
            for bbox in batch:
                assert bbox == (slice(0, 10), slice(0, 10), slice(MINT, MAXT))

    def test_random_seed(self) -> None:
        geometry = [shapely.box(0, 0, 10, 10)]
        ds = CustomGeoDataset(geometry)
        generator1 = torch.Generator().manual_seed(0)
        generator2 = torch.Generator().manual_seed(0)
        sampler1 = RandomBatchGeoSampler(ds, 1, 1, generator=generator1)
        sampler2 = RandomBatchGeoSampler(ds, 1, 1, generator=generator2)
        sample1 = next(iter(sampler1))
        sample2 = next(iter(sampler2))
        assert sample1 == sample2

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self,
        dataset: CustomGeoDataset,
        sampler: RandomBatchGeoSampler,
        num_workers: int,
    ) -> None:
        dl = DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=stack_samples,
        )
        for _ in dl:
            continue
