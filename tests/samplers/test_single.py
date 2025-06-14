# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from collections.abc import Iterator, Sequence
from datetime import datetime
from itertools import product

import pandas as pd
import pytest
import shapely
import torch
from _pytest.fixtures import SubRequest
from geopandas import GeoDataFrame
from pyproj import CRS
from shapely import Geometry, Point
from torch.utils.data import DataLoader

from torchgeo.datasets import BoundingBox, GeoDataset, stack_samples
from torchgeo.datasets.utils import GeoSlice
from torchgeo.samplers import (
    GeoSampler,
    GridGeoSampler,
    PreChippedGeoSampler,
    RandomGeoSampler,
    Units,
    tile_to_chips,
)

MINT = datetime(2025, 4, 24)
MAXT = datetime(2025, 4, 25)


class CustomGeoSampler(GeoSampler):
    def __iter__(self) -> Iterator[GeoSlice]:
        for i in range(2):
            yield slice(i, i), slice(i, i), slice(MINT, MAXT)


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


class TestGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        geometry = [shapely.box(0, 0, 100, 100)]
        return CustomGeoDataset(geometry)

    @pytest.fixture(scope='function')
    def sampler(self, dataset: CustomGeoDataset) -> CustomGeoSampler:
        return CustomGeoSampler(dataset)

    def test_iter(self, sampler: CustomGeoSampler) -> None:
        assert next(iter(sampler)) == (slice(0, 0), slice(0, 0), slice(MINT, MAXT))

    def test_abstract(self, dataset: CustomGeoDataset) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoSampler(dataset)  # type: ignore[abstract]

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: CustomGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestRandomGeoSampler:
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
    ) -> RandomGeoSampler:
        size, units = request.param
        return RandomGeoSampler(dataset, size, length=10, units=units)

    def test_iter(self, sampler: RandomGeoSampler) -> None:
        for query in sampler:
            assert sampler.roi.minx <= query.minx <= query.maxx <= sampler.roi.maxx
            assert sampler.roi.miny <= query.miny <= query.miny <= sampler.roi.maxy
            assert sampler.roi.mint <= query.mint <= query.maxt <= sampler.roi.maxt

            assert math.isclose(query.maxx - query.minx, sampler.size[1])
            assert math.isclose(query.maxy - query.miny, sampler.size[0])
            assert query.maxt - query.mint == sampler.roi.maxt - sampler.roi.mint

    def test_len(self, sampler: RandomGeoSampler) -> None:
        assert len(sampler) == sampler.length

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = BoundingBox(0, 50, 0, 50, MINT, MAXT)
        sampler = RandomGeoSampler(dataset, 2, 10, roi=roi)
        for query in sampler:
            query in roi

    def test_small_area(self) -> None:
        geometry = [shapely.box(0, 0, 10, 10), shapely.box(20, 20, 21, 21)]
        ds = CustomGeoDataset(geometry, res=(1, 1))
        sampler = RandomGeoSampler(ds, 2, 10)
        for _ in sampler:
            continue

    def test_point_data(self) -> None:
        geometry = [Point(0, 0), Point(1, 1)]
        ds = CustomGeoDataset(geometry)
        sampler = RandomGeoSampler(ds, 0, 10)
        for _ in sampler:
            continue

    def test_weighted_sampling(self) -> None:
        geometry = [shapely.box(0, 0, 0, 0), shapely.box(0, 0, 10, 10)]
        ds = CustomGeoDataset(geometry)
        sampler = RandomGeoSampler(ds, 1, 10)
        for bbox in sampler:
            assert bbox == BoundingBox(0, 10, 0, 10, MINT, MAXT)

    def test_random_seed(self) -> None:
        geometry = [shapely.box(0, 0, 10, 10)]
        ds = CustomGeoDataset(geometry)
        generator1 = torch.Generator().manual_seed(0)
        generator2 = torch.Generator().manual_seed(0)
        sampler1 = RandomGeoSampler(ds, 1, 1, generator=generator1)
        sampler2 = RandomGeoSampler(ds, 1, 1, generator=generator2)
        sample1 = next(iter(sampler1))
        sample2 = next(iter(sampler2))
        assert sample1 == sample2

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: RandomGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestGridGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        geometry = [shapely.box(0, 0, 100, 100), shapely.box(0, 0, 100, 100)]
        return CustomGeoDataset(geometry)

    @pytest.fixture(
        scope='function',
        params=product(
            [
                (8, 1),
                (6, 2),
                (4, 3),
                (4, 4),
                (2, 4),
                (2.5, 3),
                ((8, 6), (1, 2)),
                ((6, 4), (2, 3)),
                (8, None),
            ],
            [Units.PIXELS, Units.CRS],
        ),
    )
    def sampler(self, dataset: CustomGeoDataset, request: SubRequest) -> GridGeoSampler:
        (size, stride), units = request.param
        return GridGeoSampler(dataset, size, stride, units=units)

    def test_iter(self, sampler: GridGeoSampler) -> None:
        for x, y, t in sampler:
            assert (
                sampler.roi[0].start
                <= x.start
                <= x.stop
                < sampler.roi[0].stop + sampler.stride[1]
            )
            assert (
                sampler.roi[1].start
                <= y.start
                <= y.stop
                < sampler.roi[1].stop + sampler.stride[0]
            )
            assert sampler.roi[2].start <= t.start <= t.stop <= sampler.roi[2].stop

            assert math.isclose(x.stop - x.start, sampler.size[1])
            assert math.isclose(y.stop - y.start, sampler.size[0])
            assert t.stop - t.start == sampler.roi[2].stop - sampler.roi[2].start

    def test_len(self, sampler: GridGeoSampler) -> None:
        bounds = sampler.index.total_bounds
        rows, cols = tile_to_chips(bounds, sampler.size, sampler.stride)
        length = rows * cols * 2  # two items in dataset
        assert len(sampler) == length

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = BoundingBox(0, 50, 200, 250, MINT, MAXT)
        sampler = GridGeoSampler(dataset, 2, 1, roi=roi)
        for x, y, t in sampler:
            assert roi.minx <= x.start <= x.stop < roi.maxx + sampler.stride[1]
            assert roi.miny <= y.start <= y.stop < roi.maxy + sampler.stride[0]
            assert roi.mint <= t.start <= t.stop <= roi.maxt

    def test_small_area(self) -> None:
        geometry = [shapely.box(0, 0, 1, 1)]
        ds = CustomGeoDataset(geometry)
        sampler = GridGeoSampler(ds, 2, 10)
        assert len(sampler) == 0

    def test_tiles_side_by_side(self) -> None:
        geometry = [shapely.box(0, 0, 10, 10), shapely.box(0, 10, 10, 20)]
        ds = CustomGeoDataset(geometry)
        sampler = GridGeoSampler(ds, 2, 10)
        for x, y, t in sampler:
            assert x.start < x.stop
            assert y.start < y.stop

    def test_integer_multiple(self) -> None:
        geometry = [shapely.box(0, 0, 10, 10)]
        ds = CustomGeoDataset(geometry)
        sampler = GridGeoSampler(ds, 10, 10, units=Units.CRS)
        iterator = iter(sampler)
        assert len(sampler) == 1
        assert next(iterator) == (slice(0, 10), slice(0, 10), slice(MINT, MAXT))

    def test_float_multiple(self) -> None:
        geometry = [shapely.box(0, 0, 6, 5)]
        ds = CustomGeoDataset(geometry)
        sampler = GridGeoSampler(ds, 5, 5, units=Units.CRS)
        iterator = iter(sampler)
        assert len(sampler) == 2
        assert next(iterator) == (slice(0, 5), slice(0, 5), slice(MINT, MAXT))
        assert next(iterator) == (slice(5, 10), slice(0, 5), slice(MINT, MAXT))

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: GridGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestPreChippedGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        geometry = [shapely.box(0, 0, 20, 20), shapely.box(0, 0, 30, 30)]
        return CustomGeoDataset(geometry)

    @pytest.fixture(scope='function')
    def sampler(self, dataset: CustomGeoDataset) -> PreChippedGeoSampler:
        return PreChippedGeoSampler(dataset, shuffle=True)

    def test_iter(self, sampler: PreChippedGeoSampler) -> None:
        for _ in sampler:
            continue

    def test_len(self, sampler: PreChippedGeoSampler) -> None:
        assert len(sampler) == 2

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = BoundingBox(5, 15, 5, 15, MINT, MAXT)
        sampler = PreChippedGeoSampler(dataset, roi=roi)
        for x, y, t in sampler:
            assert roi.minx == x.start
            assert roi.maxx == x.stop
            assert roi.miny == y.start
            assert roi.maxy == y.stop
            assert roi.mint == t.start
            assert roi.maxt == t.stop

    def test_point_data(self) -> None:
        geometry = [shapely.Point(0, 0), shapely.Point(1, 1)]
        ds = CustomGeoDataset(geometry)
        sampler = PreChippedGeoSampler(ds)
        for _ in sampler:
            continue

    def test_shuffle_seed(self) -> None:
        geometry = [shapely.box(0, 0, 10, 10), shapely.box(0, 0, 11, 11)]
        ds = CustomGeoDataset(geometry)
        generator1 = torch.Generator().manual_seed(0)
        generator2 = torch.Generator().manual_seed(0)
        sampler1 = PreChippedGeoSampler(ds, shuffle=True, generator=generator1)
        sampler2 = PreChippedGeoSampler(ds, shuffle=True, generator=generator2)
        sample1 = next(iter(sampler1))
        sample2 = next(iter(sampler2))
        assert sample1 == sample2

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: PreChippedGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue
