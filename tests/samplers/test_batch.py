import math
from typing import Iterator, List

import pytest
from _pytest.fixtures import SubRequest
from rtree.index import Index, Property

from torchgeo.datasets import BoundingBox
from torchgeo.samplers import BatchGeoSampler, RandomBatchGeoSampler


class CustomBatchGeoSampler(BatchGeoSampler):
    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[List[BoundingBox]]:
        for i in range(len(self)):
            yield [BoundingBox(j, j, j, j, j, j) for j in range(len(self))]

    def __len__(self) -> int:
        return 2


class TestBatchGeoSampler:
    @pytest.fixture(scope="function")
    def sampler(self) -> CustomBatchGeoSampler:
        return CustomBatchGeoSampler()

    def test_iter(self, sampler: CustomBatchGeoSampler) -> None:
        expected = [BoundingBox(0, 0, 0, 0, 0, 0), BoundingBox(1, 1, 1, 1, 1, 1)]
        assert next(iter(sampler)) == expected

    def test_len(self, sampler: CustomBatchGeoSampler) -> None:
        assert len(sampler) == 2

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BatchGeoSampler(None)  # type: ignore[abstract]


class TestRandomBatchGeoSampler:
    @pytest.fixture(scope="function", params=[3, 4.5, (2, 2), (3, 4.5), (4.5, 3)])
    def sampler(self, request: SubRequest) -> RandomBatchGeoSampler:
        index = Index(interleaved=False, properties=Property(dimension=3))
        index.insert(0, (0, 10, 20, 30, 40, 50))
        index.insert(1, (0, 10, 20, 30, 40, 50))
        size = request.param
        return RandomBatchGeoSampler(index, size, batch_size=2, length=10)

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
