# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from typing import Dict, Iterator, List

import pytest
from _pytest.fixtures import SubRequest
from rasterio.crs import CRS
from torch.utils.data import DataLoader

from torchgeo.datasets import BoundingBox, GeoDataset
from torchgeo.samplers import BatchGeoSampler, RandomBatchGeoSampler


class CustomBatchGeoSampler(BatchGeoSampler):
    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        for i in range(len(self)):
            yield [BoundingBox(j, j, j, j, j, j) for j in range(len(self))]

    def __len__(self) -> int:
        return 2


class CustomGeoDataset(GeoDataset):
    def __init__(self, crs: CRS = CRS.from_epsg(3005), res: float = 1) -> None:
        super().__init__()
        self.crs = crs
        self.res = res

    def __getitem__(self, query: BoundingBox) -> Dict[str, BoundingBox]:
        return {"index": query}


class TestBatchGeoSampler:
    @pytest.fixture(scope="function")
    def sampler(self) -> CustomBatchGeoSampler:
        return CustomBatchGeoSampler()

    def test_iter(self, sampler: CustomBatchGeoSampler) -> None:
        expected = [BoundingBox(0, 0, 0, 0, 0, 0), BoundingBox(1, 1, 1, 1, 1, 1)]
        assert next(iter(sampler)) == expected

    def test_len(self, sampler: CustomBatchGeoSampler) -> None:
        assert len(sampler) == 2

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(self, sampler: CustomBatchGeoSampler, num_workers: int) -> None:
        ds = CustomGeoDataset()
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BatchGeoSampler(None)  # type: ignore[abstract]


class TestRandomBatchGeoSampler:
    @pytest.fixture(scope="function", params=[3, 4.5, (2, 2), (3, 4.5), (4.5, 3)])
    def sampler(self, request: SubRequest) -> RandomBatchGeoSampler:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 20, 30, 40, 50))
        ds.index.insert(1, (0, 10, 20, 30, 40, 50))
        size = request.param
        return RandomBatchGeoSampler(ds, size, batch_size=2, length=10)

    def test_iter(self, sampler: RandomBatchGeoSampler) -> None:
        for batch in sampler:
            for query in batch:
                assert sampler.roi.minx <= query.minx <= query.maxx <= sampler.roi.maxx
                assert sampler.roi.miny <= query.miny <= query.miny <= sampler.roi.maxy
                assert sampler.roi.mint <= query.mint <= query.maxt <= sampler.roi.maxt

                assert math.isclose(query.maxx - query.minx, sampler.size[1])
                assert math.isclose(query.maxy - query.miny, sampler.size[0])
                assert math.isclose(
                    query.maxt - query.mint, sampler.roi.maxt - sampler.roi.mint
                )

    def test_len(self, sampler: RandomBatchGeoSampler) -> None:
        assert len(sampler) == sampler.length // sampler.batch_size

    @pytest.mark.slow
    @pytest.mark.parametrize("num_workers", [0, 1, 2])
    def test_dataloader(self, sampler: RandomBatchGeoSampler, num_workers: int) -> None:
        ds = CustomGeoDataset()
        dl = DataLoader(ds, batch_sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
