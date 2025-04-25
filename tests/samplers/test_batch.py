# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
from collections.abc import Iterator
from itertools import product

import pytest
import torch
from _pytest.fixtures import SubRequest
from pyproj import CRS
from torch.utils.data import DataLoader

from torchgeo.datasets import BoundingBox, GeoDataset, stack_samples
from torchgeo.samplers import BatchGeoSampler, RandomBatchGeoSampler, Units


class CustomBatchGeoSampler(BatchGeoSampler):
    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[list[BoundingBox]]:
        for i in range(len(self)):
            yield [BoundingBox(j, j, j, j, j, j) for j in range(len(self))]

    def __len__(self) -> int:
        return 2


class CustomGeoDataset(GeoDataset):
    def __init__(
        self, crs: CRS = CRS.from_epsg(3005), res: tuple[float, float] = (10, 10)
    ) -> None:
        super().__init__()
        self._crs = crs
        self.res = res

    def __getitem__(self, query: BoundingBox) -> dict[str, BoundingBox]:
        return {'index': query}


class TestBatchGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 100, 200, 300, 400, 500))
        return ds

    @pytest.fixture(scope='function')
    def sampler(self) -> CustomBatchGeoSampler:
        return CustomBatchGeoSampler()

    def test_iter(self, sampler: CustomBatchGeoSampler) -> None:
        expected = [BoundingBox(0, 0, 0, 0, 0, 0), BoundingBox(1, 1, 1, 1, 1, 1)]
        assert next(iter(sampler)) == expected

    def test_len(self, sampler: CustomBatchGeoSampler) -> None:
        assert len(sampler) == 2

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
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 100, 200, 300, 400, 500))
        ds.index.insert(1, (0, 100, 200, 300, 400, 500))
        return ds

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

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = BoundingBox(0, 50, 200, 250, 400, 450)
        sampler = RandomBatchGeoSampler(dataset, 2, 2, 10, roi=roi)
        for batch in sampler:
            for query in batch:
                assert query in roi

    def test_small_area(self) -> None:
        ds = CustomGeoDataset(res=(1, 1))
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        ds.index.insert(1, (20, 21, 20, 21, 20, 21))
        sampler = RandomBatchGeoSampler(ds, 2, 2, 10)
        for _ in sampler:
            continue

    def test_point_data(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 0, 0, 0, 0, 0))
        ds.index.insert(1, (1, 1, 1, 1, 1, 1))
        sampler = RandomBatchGeoSampler(ds, 0, 2, 10)
        for _ in sampler:
            continue

    def test_weighted_sampling(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 0, 0, 0, 0, 0))
        ds.index.insert(1, (0, 10, 0, 10, 0, 10))
        sampler = RandomBatchGeoSampler(ds, 1, 2, 10)
        for batch in sampler:
            for bbox in batch:
                assert bbox == BoundingBox(0, 10, 0, 10, 0, 10)

    def test_random_seed(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
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
