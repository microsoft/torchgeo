# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from collections.abc import Iterator
from pathlib import Path

import pytest
import shapely
from _pytest.fixtures import SubRequest
from geopandas import GeoSeries
from matplotlib import pyplot as plt
from pandas import Interval, Timestamp
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset
from torchgeo.samplers import (
    GeoSampler,
    SpatialSampler,
    SpatioTemporalSampler,
    TemporalSampler,
)

TMIN = Timestamp(2025, 4, 1)
TMAX = Timestamp(2025, 4, 30)


class CustomGeoSampler(GeoSampler):
    def __init__(self, dataset: GeoDataset, *, length: int | None = None) -> None:
        self.hidden_length = length or 5
        if length:
            self._length = length

    def __iter__(self) -> Iterator[tuple[slice, slice, slice]]:
        for i in range(self.hidden_length):
            yield slice(i, i), slice(i, i), slice(TMIN, TMAX)


class CustomSpatialSampler(SpatialSampler):
    strategy = 'random'
    _length = 5

    def __iter__(self) -> Iterator[tuple[slice, slice]]:
        series = GeoSeries([self.geometry])
        points = series.sample_points(size=len(self)).explode()
        for point in points:
            yield slice(point.x, point.x), slice(point.y, point.y)


class CustomTemporalSampler(TemporalSampler):
    strategy = 'random'

    def _iter_subset(
        self, location: tuple[slice, slice] = (slice(None), slice(None))
    ) -> Iterator[tuple[slice, slice, slice]]:
        intervals = self._init_subset(location)
        intervals = intervals.to_series().sample(frac=1)
        x, y = location
        for interval in intervals:
            yield x, y, slice(interval.left, interval.right)


class TestGeoSampler:
    @pytest.fixture(scope='class', params=[None, 5])
    @classmethod
    def sampler(cls, dataset: GeoDataset, request: SubRequest) -> CustomGeoSampler:
        return CustomGeoSampler(dataset, length=request.param)

    def test_iter(self, sampler: CustomGeoSampler) -> None:
        assert next(iter(sampler)) == (slice(0, 0), slice(0, 0), slice(TMIN, TMAX))

    def test_len(self, sampler: CustomGeoSampler) -> None:
        assert len(sampler) == 5

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoSampler()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: CustomGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestSpatialSampler:
    @pytest.fixture(scope='class')
    @classmethod
    def sampler(cls, dataset: GeoDataset) -> CustomSpatialSampler:
        return CustomSpatialSampler(dataset)

    def test_iter(self, sampler: CustomSpatialSampler) -> None:
        x, y = next(iter(sampler))
        assert 0 <= x.start == x.stop <= 100
        assert 0 <= y.start == y.stop <= 100

    def test_plot(self, sampler: CustomSpatialSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    def test_roi(self, dataset: GeoDataset) -> None:
        roi = shapely.box(0, 0, 10, 10)
        sampler = CustomSpatialSampler(dataset, roi=roi)
        x, y = next(iter(sampler))
        assert 0 <= x.start == x.stop <= 10
        assert 0 <= y.start == y.stop <= 10

    def test_len(self, sampler: CustomSpatialSampler) -> None:
        assert len(sampler) == 5

    def test_matmul(self, dataset: GeoDataset) -> None:
        spatial_sampler = CustomSpatialSampler(dataset)
        temporal_sampler = CustomTemporalSampler(dataset)
        sampler = spatial_sampler @ temporal_sampler
        assert isinstance(sampler, SpatioTemporalSampler)

    def test_abstract(self, dataset: GeoDataset) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            SpatialSampler(dataset)

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: CustomSpatialSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestTemporalSampler:
    @pytest.fixture(scope='class')
    @classmethod
    def sampler(cls, dataset: GeoDataset) -> CustomTemporalSampler:
        return CustomTemporalSampler(dataset)

    def test_iter(self, sampler: CustomTemporalSampler) -> None:
        _, _, t = next(iter(sampler))
        assert TMIN <= t.start < t.stop <= TMAX

    def test_plot(self, sampler: CustomTemporalSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    def test_toi(self, dataset: GeoDataset) -> None:
        tmin = Timestamp(2025, 4, 10)
        tmax = Timestamp(2025, 4, 20)
        toi = Interval(tmin, tmax)
        sampler = CustomTemporalSampler(dataset, toi=toi)
        _, _, t = next(iter(sampler))
        assert tmin <= t.start < t.stop <= tmax

    def test_subset(self, sampler: CustomTemporalSampler) -> None:
        x = y = slice(0, 10)
        _, _, t = next(iter(sampler._iter_subset((x, y))))
        assert TMIN <= t.start < t.stop <= TMAX

    def test_len(self, sampler: CustomTemporalSampler) -> None:
        assert len(sampler) == 3

    def test_abstract(self, dataset: GeoDataset) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            TemporalSampler(dataset)

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: CustomTemporalSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestSpatioTemporalSampler:
    @pytest.fixture(scope='class', params=['random', 'sequential'])
    @classmethod
    def spatial_sampler(
        cls, dataset: GeoDataset, request: SubRequest
    ) -> CustomSpatialSampler:
        sampler = CustomSpatialSampler(dataset)
        sampler.strategy = request.param
        return sampler

    @pytest.fixture(scope='class', params=['random', 'sequential'])
    @classmethod
    def temporal_sampler(
        cls, dataset: GeoDataset, request: SubRequest
    ) -> CustomTemporalSampler:
        sampler = CustomTemporalSampler(dataset)
        sampler.strategy = request.param
        return sampler

    @pytest.fixture(scope='class')
    @classmethod
    def sampler(
        cls,
        spatial_sampler: CustomSpatialSampler,
        temporal_sampler: CustomTemporalSampler,
    ) -> SpatioTemporalSampler:
        return spatial_sampler @ temporal_sampler

    @pytest.mark.filterwarnings('ignore:random_sampler @ sequential_sampler')
    def test_iter(self, sampler: SpatioTemporalSampler) -> None:
        x, y, t = next(iter(sampler))
        assert 0 <= x.start == x.stop <= 100
        assert 0 <= y.start == y.stop <= 100
        assert TMIN <= t.start < t.stop <= TMAX

    def test_plot(self, sampler: SpatioTemporalSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: SpatioTemporalSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
