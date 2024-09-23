# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import os
from itertools import product

import geopandas as gpd
import pytest
from _pytest.fixtures import SubRequest
from geopandas import GeoDataFrame
from rasterio.crs import CRS
from shapely.geometry import box
from torch.utils.data import DataLoader

from torchgeo.datasets import BoundingBox, GeoDataset, stack_samples
from torchgeo.samplers import (
    GeoSampler,
    GridGeoSampler,
    PreChippedGeoSampler,
    RandomGeoSampler,
    Units,
    tile_to_chips,
)


class CustomGeoSampler(GeoSampler):
    def __init__(self) -> None:
        self.chips = self.get_chips()

    def get_chips(self) -> GeoDataFrame:
        chips = []
        for i in range(2):
            chips.append(
                {
                    'geometry': box(i, i, i, i),
                    'minx': i,
                    'miny': i,
                    'maxx': i,
                    'maxy': i,
                    'mint': i,
                    'maxt': i,
                }
            )
        return GeoDataFrame(chips, crs=CRS.from_epsg(3005))


class CustomGeoDataset(GeoDataset):
    def __init__(self, crs: CRS = CRS.from_epsg(3005), res: float = 10) -> None:
        super().__init__()
        self._crs = crs
        self.res = res

    def __getitem__(self, query: BoundingBox) -> dict[str, BoundingBox]:
        return {'index': query}


class TestGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 100, 200, 300, 400, 500))
        return ds

    @pytest.fixture(scope='function')
    def sampler(self) -> CustomGeoSampler:
        return CustomGeoSampler()

    def test_iter(self, sampler: CustomGeoSampler) -> None:
        assert next(iter(sampler)) == BoundingBox(0, 0, 0, 0, 0, 0)

    def test_len(self, sampler: CustomGeoSampler) -> None:
        assert len(sampler) == 2

    def test_abstract(self, dataset: CustomGeoDataset) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoSampler(dataset)  # type: ignore[abstract]

    @pytest.mark.parametrize(
        'filtering_file', ['filtering_4x4', 'filtering_4x4.feather']
    )
    def test_filtering_from_path(self, filtering_file: str) -> None:
        datadir = os.path.join('tests', 'data', 'samplers')
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        sampler = GridGeoSampler(
            ds, 5, 5, units=Units.CRS, roi=BoundingBox(0, 10, 0, 10, 0, 10)
        )
        iterator = iter(sampler)

        assert len(sampler) == 4
        filtering_path = os.path.join(datadir, filtering_file)
        sampler.filter_chips(filtering_path, 'intersects', 'drop')
        assert len(sampler) == 3
        assert next(iterator) == BoundingBox(5, 10, 0, 5, 0, 10)

    def test_filtering_from_gdf(self) -> None:
        datadir = os.path.join('tests', 'data', 'samplers')
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        sampler = GridGeoSampler(
            ds, 5, 5, units=Units.CRS, roi=BoundingBox(0, 10, 0, 10, 0, 10)
        )
        iterator = iter(sampler)

        # Dropping first chip
        assert len(sampler) == 4
        filtering_gdf = gpd.read_file(os.path.join(datadir, 'filtering_4x4'))
        sampler.filter_chips(filtering_gdf, 'intersects', 'drop')
        assert len(sampler) == 3
        assert next(iterator) == BoundingBox(5, 10, 0, 5, 0, 10)

        # Keeping only first chip
        sampler = GridGeoSampler(ds, 5, 5, units=Units.CRS)
        iterator = iter(sampler)
        sampler.filter_chips(filtering_gdf, 'intersects', 'keep')
        assert len(sampler) == 1
        assert next(iterator) == BoundingBox(0, 5, 0, 5, 0, 10)

    def test_set_worker_split(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        sampler = GridGeoSampler(
            ds, 5, 5, units=Units.CRS, roi=BoundingBox(0, 10, 0, 10, 0, 10)
        )
        assert len(sampler) == 4
        sampler.set_worker_split(total_workers=4, worker_num=1)
        assert len(sampler) == 1

    def test_save_chips(self, tmpdir_factory: pytest.TempdirFactory) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        sampler = GridGeoSampler(ds, 5, 5, units=Units.CRS)
        sampler.save(str(tmpdir_factory.mktemp('out').join('chips')))
        sampler.save(str(tmpdir_factory.mktemp('out').join('chips.feather')))

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: CustomGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestRandomGeoSampler:
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

    def test_empty(self, dataset: CustomGeoDataset) -> None:
        sampler = RandomGeoSampler(dataset, 5, length=0)
        assert len(sampler) == 0

    def test_refresh_samples(self, dataset: CustomGeoDataset) -> None:
        dataset.index.insert(0, (0, 100, 200, 300, 400, 500))
        sampler = RandomGeoSampler(dataset, 5, length=1)
        samples = list(sampler)
        assert len(sampler) == 1
        sampler.refresh_samples()
        assert list(sampler) != samples
        assert len(sampler) == 1

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
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: RandomGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestGridGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 100, 200, 300, 400, 500))
        ds.index.insert(1, (0, 100, 200, 300, 400, 500))
        return ds

    @pytest.fixture(
        scope='function',
        params=product(
            [
                (8, 1),
                (6, 2),
                (4, 3),
                (4, 4),
                (2, 4),
                (2.5, 3),
                ((8, 6), (1, 2)),
                ((6, 4), (2, 3)),
            ],
            [Units.PIXELS, Units.CRS],
        ),
    )
    def sampler(self, dataset: CustomGeoDataset, request: SubRequest) -> GridGeoSampler:
        (size, stride), units = request.param
        return GridGeoSampler(dataset, size, stride, units=units)

    def test_iter(self, sampler: GridGeoSampler) -> None:
        for query in sampler:
            assert (
                sampler.roi.minx
                <= query.minx
                <= query.maxx
                < sampler.roi.maxx + sampler.stride[1]
            )
            assert (
                sampler.roi.miny
                <= query.miny
                <= query.miny
                < sampler.roi.maxy + sampler.stride[0]
            )
            assert sampler.roi.mint <= query.mint <= query.maxt <= sampler.roi.maxt

            assert math.isclose(query.maxx - query.minx, sampler.size[1])
            assert math.isclose(query.maxy - query.miny, sampler.size[0])
            assert math.isclose(
                query.maxt - query.mint, sampler.roi.maxt - sampler.roi.mint
            )

    def test_len(self, sampler: GridGeoSampler) -> None:
        rows, cols = tile_to_chips(sampler.roi, sampler.size, sampler.stride)
        length = rows * cols * 2  # two items in dataset
        assert len(sampler) == length

    def test_roi(self, dataset: CustomGeoDataset) -> None:
        roi = BoundingBox(0, 50, 200, 250, 400, 450)
        sampler = GridGeoSampler(dataset, 2, 1, roi=roi)
        for query in sampler:
            assert query in roi

    def test_small_area(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 1, 0, 1, 0, 1))
        sampler = GridGeoSampler(ds, 2, 10)
        assert len(sampler) == 0

    def test_tiles_side_by_side(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        ds.index.insert(0, (0, 10, 10, 20, 0, 10))
        sampler = GridGeoSampler(ds, 2, 10)
        for bbox in sampler:
            assert bbox.area > 0

    def test_integer_multiple(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 10, 0, 10, 0, 10))
        sampler = GridGeoSampler(ds, 10, 10, units=Units.CRS)
        iterator = iter(sampler)
        assert len(sampler) == 1
        assert next(iterator) == BoundingBox(0, 10, 0, 10, 0, 10)

    def test_float_multiple(self) -> None:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 6, 0, 5, 0, 10))
        sampler = GridGeoSampler(ds, 5, 5, units=Units.CRS)
        iterator = iter(sampler)
        assert len(sampler) == 2
        assert next(iterator) == BoundingBox(0, 5, 0, 5, 0, 10)
        assert next(iterator) == BoundingBox(5, 10, 0, 5, 0, 10)

    @pytest.mark.slow
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: GridGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue


class TestPreChippedGeoSampler:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomGeoDataset:
        ds = CustomGeoDataset()
        ds.index.insert(0, (0, 20, 0, 20, 0, 20))
        ds.index.insert(1, (0, 30, 0, 30, 0, 30))
        return ds

    @pytest.fixture(scope='function')
    def sampler(self, dataset: CustomGeoDataset) -> PreChippedGeoSampler:
        return PreChippedGeoSampler(dataset, shuffle=True)

    def test_iter(self, sampler: PreChippedGeoSampler) -> None:
        for _ in sampler:
            continue

    def test_len(self, sampler: PreChippedGeoSampler) -> None:
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
    @pytest.mark.parametrize('num_workers', [0, 1, 2])
    def test_dataloader(
        self, dataset: CustomGeoDataset, sampler: PreChippedGeoSampler, num_workers: int
    ) -> None:
        dl = DataLoader(
            dataset, sampler=sampler, num_workers=num_workers, collate_fn=stack_samples
        )
        for _ in dl:
            continue
