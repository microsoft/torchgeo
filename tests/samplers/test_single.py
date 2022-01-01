# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from typing import Dict, Iterator

import pytest
from _pytest.fixtures import SubRequest
from rasterio.crs import CRS
from torch.utils.data import DataLoader

from torchgeo.datasets import BoundingBox, GeoDataset, stack_samples
from torchgeo.samplers import GeoSampler, GridGeoSampler, RandomGeoSampler


class CustomGeoSampler(GeoSampler):
    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[BoundingBox]:
        for i in range(len(self)):
            yield BoundingBox(i, i, i, i, i, i)

    def __len__(self) -> int:
        return 2


class CustomGeoDataset(GeoDataset):
    def __init__(self, crs: CRS = CRS.from_epsg(3005), res: float = 1) -> None:
        super().__init__()
        self._crs = crs
        self.res = res

    def __getitem__(self, query: BoundingBox) -> Dict[str, BoundingBox]:
        return {"index": query}


class TestGeoSampler:
    @pytest.fixture(scope="function")
    def sampler(self) -> CustomGeoSampler:
        return CustomGeoSampler()

    def test_iter(self, sampler: CustomGeoSampler) -> None:
        assert next(iter(sampler)) == BoundingBox(0, 0, 0, 0, 0, 0)

    def test_len(self, sampler: CustomGeoSampler) -> None:
        assert len(sampler) == 2

    def test_abstract(self) -> None:
        ds = CustomGeoDataset()
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoSampler(ds)  # type: ignore[abstract]

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(self, sampler: CustomGeoSampler, num_workers: int) -> None:
        ds = CustomGeoDataset()
        dl = DataLoader(
            ds, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestRandomGeoSampler:
    @pytest.fixture(scope="function", params=[3, 4.5, (2, 2), (3, 4.5), (4.5, 3)])
    def sampler(self, request: SubRequest) -> RandomGeoSampler:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 20, 30, 40, 50))
        ds.index.insert(1, (0, 10, 20, 30, 40, 50))
        size = request.param
        return RandomGeoSampler(ds, size, length=10)

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

    def test_roi(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        ds.index.insert(1, (5, 15, 5, 15, 5, 15))
        roi = BoundingBox(0, 10, 0, 10, 0, 10)
        sampler = RandomGeoSampler(ds, 2, 10, roi=roi)
        for query in sampler:
            assert query in roi

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(self, sampler: RandomGeoSampler, num_workers: int) -> None:
        ds = CustomGeoDataset()
        dl = DataLoader(ds, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestGridGeoSampler:
    @pytest.fixture(
        scope="function",
        params=[(8, 1), (6, 2), (4, 3), (2.5, 3), ((8, 6), (1, 2)), ((6, 4), (2, 3))],
    )
    def sampler(self, request: SubRequest) -> GridGeoSampler:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 20, 0, 10, 40, 50))
        ds.index.insert(1, (0, 20, 0, 10, 40, 50))
        size, stride = request.param
        return GridGeoSampler(ds, size, stride)

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
        rows = int((10 - sampler.size[0]) // sampler.stride[0]) + 1
        cols = int((20 - sampler.size[1]) // sampler.stride[1]) + 1
        length = rows * cols * 2
        assert len(sampler) == length

    def test_roi(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        ds.index.insert(1, (5, 15, 5, 15, 5, 15))
        roi = BoundingBox(0, 10, 0, 10, 0, 10)
        sampler = GridGeoSampler(ds, 2, 1, roi=roi)
        for query in sampler:
            assert query in roi

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(self, sampler: GridGeoSampler, num_workers: int) -> None:
        ds = CustomGeoDataset()
        dl = DataLoader(ds, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
