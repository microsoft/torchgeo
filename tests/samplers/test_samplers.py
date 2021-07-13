from typing import Iterator

import pytest

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
    def sampler(self) -> GeoSampler:
        return CustomGeoSampler()

    def test_iter(self, sampler: GeoSampler) -> None:
        assert next(iter(sampler)) == BoundingBox(0, 0, 0, 0, 0, 0)

    def test_len(self, sampler: GeoSampler) -> None:
        assert len(sampler) == 2

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoSampler(None)  # type: ignore[abstract]


class TestRandomGeoSampler:
    @pytest.fixture(scope="function")
    def sampler(self) -> GeoSampler:
        roi = BoundingBox(0, 10, 20, 30, 40, 50)
        return RandomGeoSampler(roi, size=5, length=10)

    def test_iter(self, sampler: GeoSampler) -> None:
        query = next(iter(sampler))

        assert sampler.roi.minx <= query.minx <= sampler.roi.maxx
        assert sampler.roi.minx <= query.maxx <= sampler.roi.maxx
        assert sampler.roi.miny <= query.miny <= sampler.roi.maxy
        assert sampler.roi.miny <= query.maxy <= sampler.roi.maxy
        assert sampler.roi.mint <= query.mint <= sampler.roi.maxt
        assert sampler.roi.mint <= query.maxt <= sampler.roi.maxt

        assert query.maxx - query.minx == sampler.size
        assert query.maxy - query.miny == sampler.size
        assert query.maxt - query.mint == sampler.roi.maxt - sampler.roi.mint

    def test_len(self, sampler: GeoSampler) -> None:
        assert len(sampler) == sampler.length
