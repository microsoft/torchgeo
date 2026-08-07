# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import math
from pathlib import Path

import pytest
import shapely
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from torchgeo.datasets import GeoDataset
from torchgeo.samplers import GriddedPatchSampler, RandomPatchSampler, Units


class TestRandomPatchSampler:
    @pytest.fixture(scope='class')
    @classmethod
    def sampler(cls, dataset: GeoDataset) -> RandomPatchSampler:
        return RandomPatchSampler(dataset, size=5)

    def test_iter(self, sampler: RandomPatchSampler) -> None:
        x, y = next(iter(sampler))
        assert 0 <= x.start < x.stop <= 100
        assert 0 <= y.start < y.stop <= 100
        assert math.isclose(x.stop - x.start, 10)
        assert math.isclose(y.stop - y.start, 10)

    def test_len(self, sampler: RandomPatchSampler) -> None:
        assert len(sampler) == 100

    def test_plot(self, sampler: RandomPatchSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    def test_length(self, dataset: GeoDataset) -> None:
        sampler = RandomPatchSampler(dataset, size=5, length=99)
        assert len(sampler) == 99

    def test_roi(self, dataset: GeoDataset) -> None:
        roi = shapely.box(0, 0, 20, 20)
        sampler = RandomPatchSampler(dataset, size=5, roi=roi)
        x, y = next(iter(sampler))
        assert 0 <= x.start < x.stop <= 20
        assert 0 <= y.start < y.stop <= 20

    def test_units(self, dataset: GeoDataset) -> None:
        sampler = RandomPatchSampler(dataset, size=10, units=Units.CRS)
        assert len(sampler) == 100

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: RandomPatchSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue


class TestGriddedPatchSampler:
    @pytest.fixture(scope='class')
    @classmethod
    def sampler(cls, dataset: GeoDataset) -> GriddedPatchSampler:
        return GriddedPatchSampler(dataset, size=5)

    def test_iter(self, sampler: GriddedPatchSampler) -> None:
        x, y = next(iter(sampler))
        assert 0 <= x.start < x.stop <= 100
        assert 0 <= y.start < y.stop <= 100
        assert math.isclose(x.stop - x.start, 10)
        assert math.isclose(y.stop - y.start, 10)

    def test_len(self, sampler: GriddedPatchSampler) -> None:
        assert len(sampler) == 100

    def test_plot(self, sampler: GriddedPatchSampler, tmp_path: Path) -> None:
        ani = sampler.plot()
        ani.save(tmp_path / 'out.gif')
        plt.close()

    def test_stride(self, dataset: GeoDataset) -> None:
        sampler = GriddedPatchSampler(dataset, size=5, stride=2.5)
        assert len(sampler) == 361

    def test_roi(self, dataset: GeoDataset) -> None:
        roi = shapely.box(0, 0, 20, 20)
        sampler = GriddedPatchSampler(dataset, size=5, roi=roi)
        x, y = next(iter(sampler))
        assert 0 <= x.start < x.stop <= 20
        assert 0 <= y.start < y.stop <= 20

    def test_units(self, dataset: GeoDataset) -> None:
        sampler = GriddedPatchSampler(dataset, size=10, units=Units.CRS)
        assert len(sampler) == 100

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: GeoDataset, sampler: GriddedPatchSampler, num_workers: int
    ) -> None:
        dl = DataLoader(dataset, sampler=sampler, num_workers=num_workers)
        for _ in dl:
            continue
