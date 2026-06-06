# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path

import pytest
from matplotlib import pyplot as plt
from pandas import Period, Timedelta, Timestamp
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset
from torchgeo.samplers import (
    RandomPeriodSampler,
    RandomTimedeltaSampler,
    RandomTimestampSampler,
    SequentialPeriodSampler,
    SequentialTimedeltaSampler,
    SequentialTimestampSampler,
)

TMIN = Timestamp(2025, 4, 1)
TMAX = Timestamp(2025, 4, 30)


class TestRandomTimestampSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> RandomTimestampSampler:
        return RandomTimestampSampler(dataset)

    def test_iter(self, sampler: RandomTimestampSampler) -> None:
        for _, _, t in sampler:
            assert TMIN <= t.start < t.stop <= TMAX
            assert t.stop - t.start == Timedelta('1D')

    def test_len(self, sampler: RandomTimestampSampler) -> None:
        assert len(sampler) == 3

    def test_plot(self, sampler: RandomTimestampSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: RandomTimestampSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestSequentialTimestampSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> SequentialTimestampSampler:
        return SequentialTimestampSampler(dataset)

    def test_iter(self, sampler: SequentialTimestampSampler) -> None:
        for _, _, t in sampler:
            assert TMIN <= t.start < t.stop <= TMAX
            assert t.stop - t.start == Timedelta('1D')

    def test_len(self, sampler: SequentialTimestampSampler) -> None:
        assert len(sampler) == 3

    def test_plot(self, sampler: SequentialTimestampSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: SequentialTimestampSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestRandomTimedeltaSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> RandomTimedeltaSampler:
        delta = Timedelta('1W')
        return RandomTimedeltaSampler(dataset, delta=delta)

    def test_iter(self, sampler: RandomTimedeltaSampler) -> None:
        for _, _, t in sampler:
            assert TMIN <= t.start < t.stop <= TMAX
            assert t.stop - t.start == Timedelta('1W')

    def test_len(self, sampler: RandomTimedeltaSampler) -> None:
        assert len(sampler) == 5

    def test_plot(self, sampler: RandomTimedeltaSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: RandomTimedeltaSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestSequentialTimedeltaSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> SequentialTimedeltaSampler:
        delta = Timedelta('1W')
        return SequentialTimedeltaSampler(dataset, delta=delta)

    def test_iter(self, sampler: SequentialTimedeltaSampler) -> None:
        for _, _, t in sampler:
            assert TMIN <= t.start < t.stop < TMAX + sampler.delta
            assert t.stop - t.start == Timedelta('1W')

    def test_len(self, sampler: SequentialTimedeltaSampler) -> None:
        assert len(sampler) == 5

    def test_plot(self, sampler: SequentialTimedeltaSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: SequentialTimedeltaSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestRandomPeriodSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> RandomPeriodSampler:
        return RandomPeriodSampler(dataset, freq='M', length=1)

    def test_iter(self, sampler: RandomPeriodSampler) -> None:
        for _, _, t in sampler:
            assert Period(t.start, freq='M') == Period('2025-4')
            assert Period(t.stop, freq='M') == Period('2025-4')

    def test_len(self, sampler: RandomPeriodSampler) -> None:
        assert len(sampler) == 1

    def test_plot(self, sampler: RandomPeriodSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: RandomPeriodSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestSequentialPeriodSampler:
    @pytest.fixture(scope='class')
    def sampler(self, dataset: GeoDataset) -> SequentialPeriodSampler:
        return SequentialPeriodSampler(dataset, freq='M')

    def test_iter(self, sampler: SequentialPeriodSampler) -> None:
        for _, _, t in sampler:
            assert Period(t.start, freq='M') == Period('2025-4')
            assert Period(t.stop, freq='M') == Period('2025-4')

    def test_len(self, sampler: SequentialPeriodSampler) -> None:
        assert len(sampler) == 1

    def test_plot(self, sampler: SequentialPeriodSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: SequentialPeriodSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
