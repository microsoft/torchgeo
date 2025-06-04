# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import os
import pickle
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import shapely
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from geopandas import GeoDataFrame
from pyproj import CRS
from rasterio.enums import Resampling
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    NAIP,
    DatasetNotFoundError,
    GeoDataset,
    IntersectionDataset,
    NonGeoClassificationDataset,
    NonGeoDataset,
    RasterDataset,
    Sentinel2,
    UnionDataset,
    VectorDataset,
)
from torchgeo.datasets.utils import GeoSlice

MINT = pd.Timestamp(2025, 4, 24)
MAXT = pd.Timestamp(2025, 4, 25)


class CustomGeoDataset(GeoDataset):
    def __init__(
        self,
        bounds: Sequence[
            tuple[float, float, float, float, pd.Timestamp, pd.Timestamp]
        ] = [(0, 1, 2, 3, MINT, MAXT)],
        crs: CRS = CRS.from_epsg(4087),
        res: float | tuple[float, float] = (1, 1),
        paths: str | os.PathLike[str] | Iterable[str | os.PathLike[str]] | None = None,
    ) -> None:
        geometry = [shapely.box(b[0], b[2], b[1], b[3]) for b in bounds]
        index = pd.IntervalIndex.from_tuples(
            [(b[4], b[5]) for b in bounds], closed='both', name='datetime'
        )
        self.index = GeoDataFrame(index=index, geometry=geometry, crs=crs)
        self.res = res
        self.paths = paths or []

    def __getitem__(self, key: GeoSlice) -> dict[str, GeoSlice]:
        xmin, xmax, xres, ymin, ymax, yres, tmin, tmax, tres = self._disambiguate_slice(
            key
        )
        interval = pd.Interval(tmin, tmax)
        index = self.index.iloc[self.index.index.overlaps(interval)]
        index = index.cx[xmin:xmax, ymin:ymax]  # type: ignore[misc]

        if index.empty:
            raise IndexError(
                f'key: {key} not found in index with bounds: {self.bounds}'
            )

        return {'index': key}


class CustomRasterDataset(RasterDataset):
    def __init__(self, dtype: torch.dtype, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._dtype = dtype

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype


class CustomVectorDataset(VectorDataset):
    filename_glob = '*.geojson'
    date_format = '%Y'
    filename_regex = r"""
        ^vector_(?P<date>\d{4})\.geojson
    """


class CustomSentinelDataset(Sentinel2):
    all_bands: tuple[str, ...] = ()
    separate_files = False


class CustomNonGeoDataset(NonGeoDataset):
    def __getitem__(self, index: int) -> dict[str, int]:
        return {'index': index}

    def __len__(self) -> int:
        return 2


class TestGeoDataset:
    @pytest.fixture
    def dataset(self) -> GeoDataset:
        return CustomGeoDataset()

    def test_getitem(self, dataset: GeoDataset) -> None:
        key = (slice(0, 1), slice(2, 3), slice(MINT, MAXT))
        assert dataset[key] == {'index': key}

    def test_len(self, dataset: GeoDataset) -> None:
        assert len(dataset) == 1

    @pytest.mark.parametrize('crs', [CRS.from_epsg(4087), CRS.from_epsg(32631)])
    def test_crs(self, dataset: GeoDataset, crs: CRS) -> None:
        dataset.crs = crs

    def test_and_two(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        dataset = ds1 & ds2
        assert isinstance(dataset, IntersectionDataset)
        assert len(dataset) == 1

    def test_and_three(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        ds3 = CustomGeoDataset()
        dataset = ds1 & ds2 & ds3
        assert isinstance(dataset, IntersectionDataset)
        assert len(dataset) == 1

    def test_and_four(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        ds3 = CustomGeoDataset()
        ds4 = CustomGeoDataset()
        dataset = (ds1 & ds2) & (ds3 & ds4)
        assert isinstance(dataset, IntersectionDataset)
        assert len(dataset) == 1

    def test_or_two(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        dataset = ds1 | ds2
        assert isinstance(dataset, UnionDataset)
        assert len(dataset) == 2

    def test_or_three(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        ds3 = CustomGeoDataset()
        dataset = ds1 | ds2 | ds3
        assert isinstance(dataset, UnionDataset)
        assert len(dataset) == 3

    def test_or_four(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        ds3 = CustomGeoDataset()
        ds4 = CustomGeoDataset()
        dataset = (ds1 | ds2) | (ds3 | ds4)
        assert isinstance(dataset, UnionDataset)
        assert len(dataset) == 4

    def test_str(self, dataset: GeoDataset) -> None:
        out = str(dataset)
        assert 'type: GeoDataset' in out
        assert 'bbox: ' in out
        assert 'size: 1' in out

    def test_picklable(self, dataset: GeoDataset) -> None:
        x = pickle.dumps(dataset)
        y = pickle.loads(x)
        assert dataset.crs == y.crs
        assert dataset.res == y.res
        assert len(dataset) == len(y)
        assert dataset.bounds == y.bounds

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoDataset()  # type: ignore[abstract]

    def test_and_nongeo(self, dataset: GeoDataset) -> None:
        ds2 = CustomNonGeoDataset()
        with pytest.raises(
            ValueError, match='IntersectionDataset only supports GeoDatasets'
        ):
            dataset & ds2  # type: ignore[operator]

    @pytest.mark.parametrize(
        'key,expected_output',
        [
            # ds[xmin:xmax:xres]
            (slice(None), (0, 1, 1, 2, 3, 1, MINT, MAXT, 1)),
            (slice(1, None), (1, 1, 1, 2, 3, 1, MINT, MAXT, 1)),
            (slice(None, 0), (0, 0, 1, 2, 3, 1, MINT, MAXT, 1)),
            (slice(None, None, -1), (0, 1, -1, 2, 3, 1, MINT, MAXT, 1)),
            (slice(1, 0, -1), (1, 0, -1, 2, 3, 1, MINT, MAXT, 1)),
            # ds[:, ymin:ymax:yres]
            ((slice(None), slice(None)), (0, 1, 1, 2, 3, 1, MINT, MAXT, 1)),
            ((slice(None), slice(1, None)), (0, 1, 1, 1, 3, 1, MINT, MAXT, 1)),
            ((slice(None), slice(None, 0)), (0, 1, 1, 2, 0, 1, MINT, MAXT, 1)),
            ((slice(None), slice(None, None, -1)), (0, 1, 1, 2, 3, -1, MINT, MAXT, 1)),
            ((slice(None), slice(1, 0, -1)), (0, 1, 1, 1, 0, -1, MINT, MAXT, 1)),
            # ds[:, :, tmin:tmax:tres]
            (
                (slice(None), slice(None), slice(None)),
                (0, 1, 1, 2, 3, 1, MINT, MAXT, 1),
            ),
            (
                (slice(None), slice(None), slice(MAXT, None)),
                (0, 1, 1, 2, 3, 1, MAXT, MAXT, 1),
            ),
            (
                (slice(None), slice(None), slice(None, MINT)),
                (0, 1, 1, 2, 3, 1, MINT, MINT, 1),
            ),
            (
                (slice(None), slice(None), slice(None, None, -1)),
                (0, 1, 1, 2, 3, 1, MINT, MAXT, -1),
            ),
            (
                (slice(None), slice(None), slice(MAXT, MINT, -1)),
                (0, 1, 1, 2, 3, 1, MAXT, MINT, -1),
            ),
            # ds[xmin:xmax:xres, ymin:ymax:yres, tmin:tmax:tres]
            (
                (slice(1, None), slice(1, None), slice(MAXT, None)),
                (1, 1, 1, 1, 3, 1, MAXT, MAXT, 1),
            ),
            (
                (slice(None, 0), slice(None, 0), slice(None, MINT)),
                (0, 0, 1, 2, 0, 1, MINT, MINT, 1),
            ),
            (
                (slice(None, None, -1), slice(None, None, -1), slice(None, None, -1)),
                (0, 1, -1, 2, 3, -1, MINT, MAXT, -1),
            ),
            (
                (slice(1, 0, -1), slice(1, 0, -1), slice(MAXT, MINT, -1)),
                (1, 0, -1, 1, 0, -1, MAXT, MINT, -1),
            ),
        ],
    )
    def test_disambiguate_slice(
        self,
        dataset: GeoDataset,
        key: GeoSlice,
        expected_output: tuple[
            float, float, float, float, float, float, pd.Timestamp, pd.Timestamp, int
        ],
    ) -> None:
        assert dataset._disambiguate_slice(key) == expected_output

    def test_files_property_for_non_existing_file_or_dir(self, tmp_path: Path) -> None:
        paths = [tmp_path, tmp_path / 'non_existing_file.tif']
        with pytest.warns(UserWarning, match='Path was ignored.'):
            assert len(CustomGeoDataset(paths=paths).files) == 0

    def test_files_property_for_virtual_files(self) -> None:
        # Tests only a subset of schemes and combinations.
        paths = [
            'file://directory/file.tif',
            'zip://archive.zip!folder/file.tif',
            'az://azure_bucket/prefix/file.tif',
            '/vsiaz/azure_bucket/prefix/file.tif',
            'zip+az://azure_bucket/prefix/archive.zip!folder_in_archive/file.tif',
            '/vsizip//vsiaz/azure_bucket/prefix/archive.zip/folder_in_archive/file.tif',
        ]
        assert len(CustomGeoDataset(paths=paths).files) == len(paths)

    def test_files_property_ordered(self) -> None:
        """Ensure that the list of files is ordered."""
        paths = ['file://file3.tif', 'file://file1.tif', 'file://file2.tif']
        assert CustomGeoDataset(paths=paths).files == sorted(paths)

    def test_files_property_deterministic(self) -> None:
        """Ensure that the list of files is consistent regardless of their original
        order.
        """
        paths1 = ['file://file3.tif', 'file://file1.tif', 'file://file2.tif']
        paths2 = ['file://file2.tif', 'file://file3.tif', 'file://file1.tif']
        assert (
            CustomGeoDataset(paths=paths1).files == CustomGeoDataset(paths=paths2).files
        )

    def test_files_property_mix_str_and_pathlib(self, tmp_path: Path) -> None:
        foo = tmp_path / 'foo.txt'
        bar = tmp_path / 'bar.txt'
        foo.touch()
        bar.touch()
        ds = CustomGeoDataset(paths=[str(foo), bar])
        assert ds.files == [str(bar), str(foo)]


class TestRasterDataset:
    naip_dir = os.path.join('tests', 'data', 'naip')
    s2_dir = os.path.join(
        'tests',
        'data',
        'sentinel2',
        'S2A_MSIL2A_20220414T110751_N0400_R108_T26EMU_20220414T165533.SAFE',
        'GRANULE',
        'L2A_T26EMU_A035569_20220414T110747',
        'IMG_DATA',
        'R10m',
    )

    @pytest.fixture(params=zip([['R', 'G', 'B'], None], [True, False]))
    def naip(self, request: SubRequest) -> NAIP:
        bands = request.param[0]
        crs = CRS.from_epsg(4087)
        transforms = nn.Identity()
        cache = request.param[1]
        return NAIP(
            self.naip_dir, crs=crs, bands=bands, transforms=transforms, cache=cache
        )

    @pytest.fixture(
        params=zip(
            [
                ['B04', 'B03', 'B02'],
                ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B09', 'B11'],
            ],
            [True, False],
        )
    )
    def sentinel(self, request: SubRequest) -> Sentinel2:
        root = os.path.join('tests', 'data', 'sentinel2')
        bands = request.param[0]
        transforms = nn.Identity()
        cache = request.param[1]
        return Sentinel2(root, bands=bands, transforms=transforms, cache=cache)

    @pytest.mark.parametrize(
        'paths',
        [
            # Single directory
            naip_dir,
            # Multiple directories
            [naip_dir, naip_dir],
            # Multiple files
            (
                os.path.join(naip_dir, 'm_3807511_ne_18_060_20181104.tif'),
                os.path.join(naip_dir, 'm_3807511_ne_18_060_20190605.tif'),
            ),
            # Combination
            {naip_dir, os.path.join(naip_dir, 'm_3807511_ne_18_060_20181104.tif')},
        ],
    )
    def test_files(self, paths: str | Iterable[str]) -> None:
        assert len(NAIP(paths).files) == 2

    @pytest.mark.parametrize(
        'paths',
        [
            # Single directory
            s2_dir,
            # Multiple directories
            [s2_dir, s2_dir],
            # Multiple files (single band)
            [
                os.path.join(s2_dir, 'T26EMU_20190414T110751_B04_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20220414T110751_B04_10m.jp2'),
            ],
            # Multiple files (multiple bands)
            [
                os.path.join(s2_dir, 'T26EMU_20190414T110751_B04_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20190414T110751_B03_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20190414T110751_B02_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20220414T110751_B04_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20220414T110751_B03_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20220414T110751_B02_10m.jp2'),
            ],
            # Combination
            [
                s2_dir,
                os.path.join(s2_dir, 'T26EMU_20190414T110751_B04_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20220414T110751_B04_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20220414T110751_B03_10m.jp2'),
                os.path.join(s2_dir, 'T26EMU_20220414T110751_B02_10m.jp2'),
            ],
        ],
    )
    @pytest.mark.filterwarnings('ignore:Could not find any relevant files')
    def test_files_separate(self, paths: str | Iterable[str]) -> None:
        assert len(Sentinel2(paths, bands=Sentinel2.rgb_bands).files) == 2

    def test_getitem_single_file(self, naip: NAIP) -> None:
        xmin, xmax, ymin, ymax, tmin, tmax = naip.bounds
        x = naip[xmin:xmax, ymin:ymax, tmin:tmax]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)
        assert len(naip.bands) == x['image'].shape[0]

    def test_getitem_separate_files(self, sentinel: Sentinel2) -> None:
        xmin, xmax, ymin, ymax, tmin, tmax = sentinel.bounds
        x = sentinel[xmin:xmax, ymin:ymax, tmin:tmax]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)
        assert len(sentinel.bands) == x['image'].shape[0]

    def test_reprojection(self, naip: NAIP) -> None:
        naip2 = NAIP(naip.paths, crs=CRS.from_epsg(4326))
        assert naip.crs != naip2.crs
        assert not math.isclose(naip.res[0], naip2.res[0])
        assert not math.isclose(naip.res[1], naip2.res[1])

    @pytest.mark.parametrize('dtype', ['uint16', 'uint32'])
    def test_getitem_uint_dtype(self, dtype: str) -> None:
        root = os.path.join('tests', 'data', 'raster', dtype)
        ds = RasterDataset(root)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        x = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].dtype == torch.float32

    @pytest.mark.parametrize('dtype', [torch.float, torch.double])
    def test_resampling_float_dtype(self, dtype: torch.dtype) -> None:
        paths = os.path.join('tests', 'data', 'raster', 'uint16')
        ds = CustomRasterDataset(dtype, paths)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        x = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert x['image'].dtype == dtype
        assert ds.resampling == Resampling.bilinear

    @pytest.mark.parametrize('dtype', [torch.long, torch.bool])
    def test_resampling_int_dtype(self, dtype: torch.dtype) -> None:
        paths = os.path.join('tests', 'data', 'raster', 'uint16')
        ds = CustomRasterDataset(dtype, paths)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        x = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert x['image'].dtype == dtype
        assert ds.resampling == Resampling.nearest

    def test_invalid_key(self, sentinel: Sentinel2) -> None:
        with pytest.raises(
            IndexError, match='key: .* not found in index with bounds: .*'
        ):
            sentinel[0:0, 0:0, pd.Timestamp.min : pd.Timestamp.min]

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            RasterDataset(tmp_path)

    def test_no_all_bands(self) -> None:
        root = os.path.join('tests', 'data', 'sentinel2')
        bands = ('B04', 'B03', 'B02')
        transforms = nn.Identity()
        cache = True
        msg = (
            'CustomSentinelDataset is missing an `all_bands` attribute,'
            ' so `bands` cannot be specified.'
        )

        with pytest.raises(AssertionError, match=msg):
            CustomSentinelDataset(root, bands=bands, transforms=transforms, cache=cache)

    def test_single_res(self) -> None:
        root = os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_32631')
        ds = RasterDataset(root, res=10.0)
        assert ds.res == (10.0, 10.0)

    def test_change_res(self) -> None:
        root = os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_32631')
        ds = RasterDataset(root, res=10.0)
        assert ds.res == (10.0, 10.0)
        ds.res = 20.0


class TestVectorDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomVectorDataset:
        root = os.path.join('tests', 'data', 'vector')
        transforms = nn.Identity()
        return CustomVectorDataset(root, res=(0.1, 0.1), transforms=transforms)

    @pytest.fixture(scope='class')
    def multilabel(self) -> CustomVectorDataset:
        root = os.path.join('tests', 'data', 'vector')
        transforms = nn.Identity()
        return CustomVectorDataset(
            root, res=(0.1, 0.1), transforms=transforms, label_name='label_id'
        )

    def test_getitem(self, dataset: CustomVectorDataset) -> None:
        xmin, xmax, ymin, ymax, tmin, tmax = dataset.bounds
        x = dataset[xmin:xmax, ymin:ymax, tmin:tmax]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(
            x['mask'].unique(),  # type: ignore[no-untyped-call]
            torch.tensor([0, 1], dtype=torch.uint8),
        )

    def test_time_index(self, dataset: CustomVectorDataset) -> None:
        assert dataset.bounds[4] > pd.Timestamp.min
        assert dataset.bounds[5] < pd.Timestamp.max

    def test_getitem_multilabel(self, multilabel: CustomVectorDataset) -> None:
        xmin, xmax, ymin, ymax, tmin, tmax = multilabel.bounds
        x = multilabel[xmin:xmax, ymin:ymax, tmin:tmax]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(
            x['mask'].unique(),  # type: ignore[no-untyped-call]
            torch.tensor([0, 1, 2, 3], dtype=torch.uint8),
        )

    def test_empty_shapes(self, dataset: CustomVectorDataset) -> None:
        x = dataset[1.1:1.9, 1.1:1.9, pd.Timestamp.min : pd.Timestamp.max]
        assert torch.equal(x['mask'], torch.zeros(8, 8, dtype=torch.uint8))

    def test_invalid_key(self, dataset: CustomVectorDataset) -> None:
        with pytest.raises(IndexError, match='key: .* not found in index with bounds:'):
            dataset[3:3, 3:3, pd.Timestamp.min : pd.Timestamp.min]

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            VectorDataset(tmp_path)

    def test_single_res(self) -> None:
        root = os.path.join('tests', 'data', 'vector')
        ds = CustomVectorDataset(root, res=0.1)
        assert ds.res == (0.1, 0.1)


class TestNonGeoDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> NonGeoDataset:
        return CustomNonGeoDataset()

    def test_getitem(self, dataset: NonGeoDataset) -> None:
        assert dataset[0] == {'index': 0}

    def test_len(self, dataset: NonGeoDataset) -> None:
        assert len(dataset) == 2

    def test_add_two(self) -> None:
        ds1 = CustomNonGeoDataset()
        ds2 = CustomNonGeoDataset()
        dataset = ds1 + ds2
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 4

    def test_add_three(self) -> None:
        ds1 = CustomNonGeoDataset()
        ds2 = CustomNonGeoDataset()
        ds3 = CustomNonGeoDataset()
        dataset = ds1 + ds2 + ds3
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 6

    def test_add_four(self) -> None:
        ds1 = CustomNonGeoDataset()
        ds2 = CustomNonGeoDataset()
        ds3 = CustomNonGeoDataset()
        ds4 = CustomNonGeoDataset()
        dataset = (ds1 + ds2) + (ds3 + ds4)
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 8

    def test_str(self, dataset: NonGeoDataset) -> None:
        assert 'type: NonGeoDataset' in str(dataset)
        assert 'size: 2' in str(dataset)

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            NonGeoDataset()  # type: ignore[abstract]


class TestNonGeoClassificationDataset:
    @pytest.fixture(scope='class')
    def dataset(self, root: str) -> NonGeoClassificationDataset:
        transforms = nn.Identity()
        return NonGeoClassificationDataset(root, transforms=transforms)

    @pytest.fixture(scope='class')
    def root(self) -> str:
        root = os.path.join('tests', 'data', 'nongeoclassification')
        return root

    def test_getitem(self, dataset: NonGeoClassificationDataset) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert isinstance(x['label'], torch.Tensor)
        assert x['image'].shape[0] == 3

    def test_len(self, dataset: NonGeoClassificationDataset) -> None:
        assert len(dataset) == 2

    def test_add_two(self, root: str) -> None:
        ds1 = NonGeoClassificationDataset(root)
        ds2 = NonGeoClassificationDataset(root)
        dataset = ds1 + ds2
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 4

    def test_add_three(self, root: str) -> None:
        ds1 = NonGeoClassificationDataset(root)
        ds2 = NonGeoClassificationDataset(root)
        ds3 = NonGeoClassificationDataset(root)
        dataset = ds1 + ds2 + ds3
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 6

    def test_add_four(self, root: str) -> None:
        ds1 = NonGeoClassificationDataset(root)
        ds2 = NonGeoClassificationDataset(root)
        ds3 = NonGeoClassificationDataset(root)
        ds4 = NonGeoClassificationDataset(root)
        dataset = (ds1 + ds2) + (ds3 + ds4)
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 8

    def test_str(self, dataset: NonGeoClassificationDataset) -> None:
        assert 'type: NonGeoDataset' in str(dataset)
        assert 'size: 2' in str(dataset)


class TestIntersectionDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> IntersectionDataset:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_4-4_epsg_4326')
        )
        transforms = nn.Identity()
        return IntersectionDataset(ds1, ds2, transforms=transforms)

    def test_getitem(self, dataset: IntersectionDataset) -> None:
        xmin, xmax, ymin, ymax, tmin, tmax = dataset.bounds
        sample = dataset[xmin:xmax, ymin:ymax, tmin:tmax]
        assert isinstance(sample['image'], torch.Tensor)

    def test_len(self, dataset: IntersectionDataset) -> None:
        assert len(dataset) == 1

    def test_str(self, dataset: IntersectionDataset) -> None:
        out = str(dataset)
        assert 'type: IntersectionDataset' in out
        assert 'bbox: ' in out
        assert 'size: 1' in out

    def test_nongeo_dataset(self) -> None:
        ds1 = CustomNonGeoDataset()
        ds2 = CustomNonGeoDataset()
        with pytest.raises(
            ValueError, match='IntersectionDataset only supports GeoDatasets'
        ):
            IntersectionDataset(ds1, ds2)  # type: ignore[arg-type]

    def test_multiple_res_12(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-1_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds = IntersectionDataset(ds1, ds2)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == (2, 1)
        assert len(ds1) == len(ds2) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_multiple_res_21(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-1_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds = IntersectionDataset(ds2, ds1)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_12(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4326')
        )
        ds = IntersectionDataset(ds1, ds2)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_12_3(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4326')
        )
        ds3 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_32631')
        )
        ds = (ds1 & ds2) & ds3
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds3) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_1_23(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4326')
        )
        ds3 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_32631')
        )
        ds = ds1 & (ds2 & ds3)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds3) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_12(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_4-4_epsg_4087')
        )
        ds = IntersectionDataset(ds1, ds2)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_12_3(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_4-4_epsg_4087')
        )
        ds3 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_8-8_epsg_4087')
        )
        ds = (ds1 & ds2) & ds3
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds3) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_1_23(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_4-4_epsg_4087')
        )
        ds3 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_8-8_epsg_4087')
        )
        ds = ds1 & (ds2 & ds3)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds3) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_single_res(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-1_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds = IntersectionDataset(ds1, ds2)
        ds.res = 10
        assert ds1.res == ds2.res == ds.res == (10, 10)

    def test_spatial_intersection(self) -> None:
        bounds1 = [
            (0, 2, 0, 2, MINT, MAXT),
            (1, 3, 1, 3, MINT, MAXT),
            (2, 4, 2, 4, MINT, MAXT),
            (4, 6, 4, 6, MINT, MAXT),
        ]
        bounds2 = [(1, 3, 1, 3, MINT, MAXT)]
        ds1 = CustomGeoDataset(bounds1)
        ds2 = CustomGeoDataset(bounds2)
        ds = IntersectionDataset(ds1, ds2)
        assert len(ds) == 3
        assert shapely.box(1, 1, 2, 2) in ds.index.geometry
        assert shapely.box(1, 1, 3, 3) in ds.index.geometry
        assert shapely.box(2, 2, 3, 3) in ds.index.geometry

    def test_temporal_intersection(self) -> None:
        bounds1 = [
            (0, 1, 0, 1, pd.Timestamp(2025, 4, 1), pd.Timestamp(2025, 4, 3)),
            (0, 1, 0, 1, pd.Timestamp(2025, 4, 7), pd.Timestamp(2025, 4, 9)),
            (0, 1, 0, 1, pd.Timestamp(2025, 4, 4), pd.Timestamp(2025, 4, 6)),
        ]
        bounds2 = [
            (0, 1, 0, 1, pd.Timestamp(2025, 5, 1), pd.Timestamp(2025, 5, 9)),
            (0, 1, 0, 1, pd.Timestamp(2025, 4, 2), pd.Timestamp(2025, 4, 5)),
        ]
        ds1 = CustomGeoDataset(bounds1)
        ds2 = CustomGeoDataset(bounds2)
        ds = IntersectionDataset(ds1, ds2)
        assert len(ds) == 2
        assert ds.index.index[0].left == pd.Timestamp(2025, 4, 2)
        assert ds.index.index[0].right == pd.Timestamp(2025, 4, 3)
        assert ds.index.index[1].left == pd.Timestamp(2025, 4, 4)
        assert ds.index.index[1].right == pd.Timestamp(2025, 4, 5)

    def test_spatiotemporal_intersection(self) -> None:
        bounds1 = [
            (0, 2, 0, 2, pd.Timestamp(2025, 4, 1), pd.Timestamp(2025, 4, 3)),
            (1, 3, 1, 3, pd.Timestamp(2025, 4, 7), pd.Timestamp(2025, 4, 9)),
            (2, 4, 2, 4, pd.Timestamp(2025, 4, 4), pd.Timestamp(2025, 4, 6)),
        ]
        bounds2 = [
            (1, 3, 1, 3, pd.Timestamp(2025, 4, 2), pd.Timestamp(2025, 4, 5)),
            (1, 3, 1, 3, pd.Timestamp(2025, 5, 1), pd.Timestamp(2025, 5, 9)),
            (5, 6, 5, 6, pd.Timestamp(2025, 4, 2), pd.Timestamp(2025, 4, 5)),
            (5, 6, 5, 6, pd.Timestamp(2025, 5, 1), pd.Timestamp(2025, 5, 9)),
        ]
        ds1 = CustomGeoDataset(bounds1)
        ds2 = CustomGeoDataset(bounds2)
        ds = IntersectionDataset(ds1, ds2)
        assert len(ds) == 2
        assert shapely.box(1, 1, 2, 2) in ds.index.geometry
        assert shapely.box(2, 2, 3, 3) in ds.index.geometry
        assert ds.index.index[0].left == pd.Timestamp(2025, 4, 2)
        assert ds.index.index[0].right == pd.Timestamp(2025, 4, 3)
        assert ds.index.index[1].left == pd.Timestamp(2025, 4, 4)
        assert ds.index.index[1].right == pd.Timestamp(2025, 4, 5)

    def test_point_dataset(self) -> None:
        ds1 = CustomGeoDataset([(0, 2, 2, 4, MINT, MAXT)])
        ds2 = CustomGeoDataset([(1, 1, 3, 3, MINT, MINT)])
        msg = 'Datasets have no spatiotemporal intersection'
        with pytest.raises(RuntimeError, match=msg):
            IntersectionDataset(ds1, ds2)

    def test_no_overlap(self) -> None:
        ds1 = CustomGeoDataset([(0, 1, 2, 3, MINT, MINT)])
        ds2 = CustomGeoDataset([(6, 7, 8, 9, MAXT, MAXT)])
        msg = 'Datasets have no spatiotemporal intersection'
        with pytest.raises(RuntimeError, match=msg):
            IntersectionDataset(ds1, ds2)

    def test_grid_overlap(self) -> None:
        ds1 = CustomGeoDataset([(0, 1, 2, 3, MINT, MAXT)])
        ds2 = CustomGeoDataset([(1, 2, 3, 4, MAXT, MAXT)])
        msg = 'Datasets have no spatiotemporal intersection'
        with pytest.raises(RuntimeError, match=msg):
            IntersectionDataset(ds1, ds2)

    def test_invalid_key(self, dataset: IntersectionDataset) -> None:
        with pytest.raises(IndexError, match='key: .* not found in index with bounds:'):
            dataset[-1:-1, -1:-1, pd.Timestamp.min : pd.Timestamp.min]


class TestUnionDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> UnionDataset:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_4-4_epsg_4326')
        )
        transforms = nn.Identity()
        return UnionDataset(ds1, ds2, transforms=transforms)

    def test_getitem(self, dataset: UnionDataset) -> None:
        xmin, xmax, ymin, ymax, tmin, tmax = dataset.bounds
        sample = dataset[xmin:xmax, ymin:ymax, tmin:tmax]
        assert isinstance(sample['image'], torch.Tensor)

    def test_len(self, dataset: UnionDataset) -> None:
        assert len(dataset) == 2

    def test_str(self, dataset: UnionDataset) -> None:
        out = str(dataset)
        assert 'type: UnionDataset' in out
        assert 'bbox: ' in out
        assert 'size: 2' in out

    def test_different_crs_12(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4326')
        )
        ds = UnionDataset(ds1, ds2)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == 1
        assert len(ds) == 2
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_12_3(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4326')
        )
        ds3 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_32631')
        )
        ds = (ds1 | ds2) | ds3
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds3) == 1
        assert len(ds) == 3
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_1_23(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4326')
        )
        ds3 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_32631')
        )
        ds = ds1 | (ds2 | ds3)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds3) == 1
        assert len(ds) == 3
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_12(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_4-4_epsg_4087')
        )
        ds = UnionDataset(ds1, ds2)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == 1
        assert len(ds) == 2
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_12_3(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_4-4_epsg_4087')
        )
        ds3 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_8-8_epsg_4087')
        )
        ds = (ds1 | ds2) | ds3
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds3) == 1
        assert len(ds) == 3
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_1_23(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_4-4_epsg_4087')
        )
        ds3 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_8-8_epsg_4087')
        )
        ds = ds1 | (ds2 | ds3)
        xmin, xmax, ymin, ymax, tmin, tmax = ds.bounds
        sample = ds[xmin:xmax, ymin:ymax, tmin:tmax]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == (2, 2)
        assert len(ds1) == len(ds2) == len(ds3) == 1
        assert len(ds) == 3
        assert isinstance(sample['image'], torch.Tensor)

    def test_no_overlap(self) -> None:
        ds1 = CustomGeoDataset([(0, 1, 0, 1, MINT, MAXT)])
        ds2 = CustomGeoDataset([(2, 3, 2, 3, MINT, MAXT)])
        ds = UnionDataset(ds1, ds2)
        ds[0:1, 0:1, MINT:MAXT]
        ds[2:3, 2:3, MINT:MAXT]

    def test_single_res(self) -> None:
        ds1 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-1_epsg_4087')
        )
        ds2 = RasterDataset(
            os.path.join('tests', 'data', 'raster', 'res_2-2_epsg_4087')
        )
        ds = UnionDataset(ds1, ds2)
        ds.res = 10
        assert ds1.res == ds2.res == ds.res == (10, 10)

    def test_nongeo_dataset(self) -> None:
        ds1 = CustomNonGeoDataset()
        ds2 = CustomNonGeoDataset()
        ds3 = CustomGeoDataset()
        msg = 'UnionDataset only supports GeoDatasets'
        with pytest.raises(ValueError, match=msg):
            UnionDataset(ds1, ds2)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=msg):
            UnionDataset(ds1, ds3)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match=msg):
            UnionDataset(ds3, ds1)  # type: ignore[arg-type]

    def test_invalid_key(self, dataset: UnionDataset) -> None:
        with pytest.raises(IndexError, match='key: .* not found in index with bounds:'):
            dataset[-1:-1, -1:-1, pd.Timestamp.min : pd.Timestamp.min]
