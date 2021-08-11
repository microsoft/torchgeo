import math
from typing import Iterator

import pytest
from _pytest.fixtures import SubRequest
from rtree.index import Index, Property

from torchgeo.datasets import BoundingBox
from torchgeo.samplers import GeoSampler, GridGeoSampler, RandomGeoSampler


class CustomGeoSampler(GeoSampler):
    def __init__(self) -> None:
        pass

    def __iter__(self) -> Iterator[BoundingBox]:
        for i in range(len(self)):
            yield BoundingBox(i, i, i, i, i, i)

    def __len__(self) -> int:
        return 2


class TestGeoSampler:
    @pytest.fixture(scope="function")
    def sampler(self) -> CustomGeoSampler:
        return CustomGeoSampler()

    def test_iter(self, sampler: CustomGeoSampler) -> None:
        assert next(iter(sampler)) == BoundingBox(0, 0, 0, 0, 0, 0)

    def test_len(self, sampler: CustomGeoSampler) -> None:
        assert len(sampler) == 2

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoSampler(None)  # type: ignore[abstract]


class TestRandomGeoSampler:
    @pytest.fixture(scope="function", params=[3, 4.5, (2, 2), (3, 4.5), (4.5, 3)])
    def sampler(self, request: SubRequest) -> RandomGeoSampler:
        index = Index(interleaved=False, properties=Property(dimension=3))
        index.insert(0, (0, 10, 20, 30, 40, 50))
        index.insert(1, (0, 10, 20, 30, 40, 50))
        size = request.param
        return RandomGeoSampler(index, size, length=10)

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


class TestGridGeoSampler:
    @pytest.fixture(
        scope="function",
        params=[
            (8, 1),
            (6, 2),
            (4, 3),
            (2.5, 3),
            ((8, 6), (1, 2)),
            ((6, 4), (2, 3)),
        ],
    )
    def sampler(self, request: SubRequest) -> GridGeoSampler:
        index = Index(interleaved=False, properties=Property(dimension=3))
        index.insert(0, (0, 10, 20, 30, 40, 50))
        index.insert(1, (0, 10, 20, 30, 40, 50))
        size, stride = request.param
        return GridGeoSampler(index, size, stride)

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
