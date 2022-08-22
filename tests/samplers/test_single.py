# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from itertools import product
from typing import Dict, Iterator

import pytest
from _pytest.fixtures import SubRequest
from rasterio.crs import CRS
from torch.utils.data import DataLoader

from torchgeo.datasets import BoundingBox, GeoDataset, stack_samples
from torchgeo.samplers import (
    GeoSampler,
    GridGeoSampler,
    PreChippedGeoSampler,
    RandomGeoSampler,
    Units,
)


class CustomGeoSampler(GeoSampler):
    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[BoundingBox]:
        for i in range(len(self)):
            yield BoundingBox(i, i, i, i, i, i)

    def __len__(self) -> int:
        return 2


class CustomGeoDataset(GeoDataset):
    def __init__(self, crs: CRS = CRS.from_epsg(3005), res: float = 10) -> None:
        super().__init__()
        self._crs = crs
        self.res = res

    def __getitem__(self, query: BoundingBox) -> Dict[str, BoundingBox]:
        return {"index": query}


class TestGeoSampler:
    @pytest.fixture(scope="class")
    def dataset(self) -> CustomGeoDataset:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 100, 200, 300, 400, 500))
        return ds

    @pytest.fixture(scope="function")
    def sampler(self) -> CustomGeoSampler:
        return CustomGeoSampler()

    def test_iter(self, sampler: CustomGeoSampler) -> None:
        assert next(iter(sampler)) == BoundingBox(0, 0, 0, 0, 0, 0)

    def test_len(self, sampler: CustomGeoSampler) -> None:
        assert len(sampler) == 2

    def test_abstract(self, dataset: CustomGeoDataset) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoSampler(dataset)  # type: ignore[abstract]

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: CustomGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestRandomGeoSampler:
    @pytest.fixture(scope="class")
    def dataset(self) -> CustomGeoDataset:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 100, 200, 300, 400, 500))
        ds.index.insert(1, (0, 100, 200, 300, 400, 500))
        return ds

    @pytest.fixture(
        scope="function",
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
            assert math.isclose(
                query.maxt - query.mint, sampler.roi.maxt - sampler.roi.mint
            )

    def test_len(self, sampler: RandomGeoSampler) -> None:
        assert len(sampler) == sampler.length

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = BoundingBox(0, 50, 200, 250, 400, 450)
        sampler = RandomGeoSampler(dataset, 2, 10, roi=roi)
        for query in sampler:
            assert query in roi

    def test_small_area(self) -> None:
        ds = CustomGeoDataset(res=1)
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        ds.index.insert(1, (20, 21, 20, 21, 20, 21))
        sampler = RandomGeoSampler(ds, 2, 10)
        for _ in sampler:
            continue

    def test_point_data(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 0, 0, 0, 0, 0))
        ds.index.insert(1, (1, 1, 1, 1, 1, 1))
        sampler = RandomGeoSampler(ds, 0, 10)
        for _ in sampler:
            continue

    def test_weighted_sampling(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 0, 0, 0, 0, 0))
        ds.index.insert(1, (0, 10, 0, 10, 0, 10))
        sampler = RandomGeoSampler(ds, 1, 10)
        for bbox in sampler:
            assert bbox == BoundingBox(0, 10, 0, 10, 0, 10)

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: RandomGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestGridGeoSampler:
    @pytest.fixture(scope="class")
    def dataset(self) -> CustomGeoDataset:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 100, 200, 300, 400, 500))
        ds.index.insert(1, (0, 100, 200, 300, 400, 500))
        return ds

    @pytest.fixture(
        scope="function",
        params=product(
            [(8, 1), (6, 2), (4, 3), (2.5, 3), ((8, 6), (1, 2)), ((6, 4), (2, 3))],
            [Units.PIXELS, Units.CRS],
        ),
    )
    def sampler(self, dataset: CustomGeoDataset, request: SubRequest) -> GridGeoSampler:
        (size, stride), units = request.param
        return GridGeoSampler(dataset, size, stride, units=units)

    def test_iter(self, sampler: GridGeoSampler) -> None:
        for query in sampler:
            assert sampler.roi.minx <= query.minx <= query.maxx <= sampler.roi.maxx
            assert sampler.roi.miny <= query.miny <= query.miny <= sampler.roi.maxy
            assert sampler.roi.mint <= query.mint <= query.maxt <= sampler.roi.maxt

            assert math.isclose(query.maxx - query.minx, sampler.size[1])
            assert math.isclose(query.maxy - query.miny, sampler.size[0])
            assert math.isclose(
                query.maxt - query.mint, sampler.roi.maxt - sampler.roi.mint
            )

    def test_len(self, sampler: GridGeoSampler) -> None:
        rows = ((100 - sampler.size[0]) // sampler.stride[0]) + 1
        cols = ((100 - sampler.size[1]) // sampler.stride[1]) + 1
        length = rows * cols * 2
        assert len(sampler) == length

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = BoundingBox(0, 50, 200, 250, 400, 450)
        sampler = GridGeoSampler(dataset, 2, 1, roi=roi)
        for query in sampler:
            assert query in roi

    def test_small_area(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        ds.index.insert(1, (20, 21, 20, 21, 20, 21))
        sampler = GridGeoSampler(ds, 2, 10)
        for _ in sampler:
            continue

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: GridGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestPreChippedGeoSampler:
    @pytest.fixture(scope="class")
    def dataset(self) -> CustomGeoDataset:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 20, 0, 20, 0, 20))
        ds.index.insert(1, (0, 30, 0, 30, 0, 30))
        return ds

    @pytest.fixture(scope="function")
    def sampler(self, dataset: CustomGeoDataset) -> PreChippedGeoSampler:
        return PreChippedGeoSampler(dataset, shuffle=True)

    def test_iter(self, sampler: GridGeoSampler) -> None:
        for _ in sampler:
            continue

    def test_len(self, sampler: GridGeoSampler) -> None:
        assert len(sampler) == 2

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = BoundingBox(5, 15, 5, 15, 5, 15)
        sampler = PreChippedGeoSampler(dataset, roi=roi)
        for query in sampler:
            assert query == roi

    def test_point_data(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 0, 0, 0, 0, 0))
        ds.index.insert(1, (1, 1, 1, 1, 1, 1))
        sampler = PreChippedGeoSampler(ds)
        for _ in sampler:
            continue

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: PreChippedGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue
