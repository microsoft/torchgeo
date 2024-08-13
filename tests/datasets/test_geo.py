# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import math
import os
import pickle
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from rasterio.crs import CRS
from rasterio.enums import Resampling
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    NAIP,
    BoundingBox,
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


class CustomGeoDataset(GeoDataset):
    def __init__(
        self,
        bounds: BoundingBox = BoundingBox(0, 1, 2, 3, 4, 5),
        crs: CRS = CRS.from_epsg(4087),
        res: float = 1,
        paths: str | Path | Iterable[str | Path] | None = None,
    ) -> None:
        super().__init__()
        self.index.insert(0, tuple(bounds))
        self._crs = crs
        self.res = res
        self.paths = paths or []

    def __getitem__(self, query: BoundingBox) -> dict[str, BoundingBox]:
        hits = self.index.intersection(tuple(query), objects=True)
        hit = next(iter(hits))
        bounds = BoundingBox(*hit.bounds)
        return {'index': bounds}


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
    all_bands: list[str] = []
    separate_files = False


class CustomNonGeoDataset(NonGeoDataset):
    def __getitem__(self, index: int) -> dict[str, int]:
        return {'index': index}

    def __len__(self) -> int:
        return 2


class TestGeoDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> GeoDataset:
        return CustomGeoDataset()

    def test_getitem(self, dataset: GeoDataset) -> None:
        query = BoundingBox(0, 1, 2, 3, 4, 5)
        assert dataset[query] == {'index': query}

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
        assert 'bbox: BoundingBox' in out
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
    @pytest.mark.filterwarnings("ignore:Could not find any relevant files")
    def test_files_separate(self, paths: str | Iterable[str]) -> None:
        assert len(Sentinel2(paths, bands=Sentinel2.rgb_bands).files) == 2

    def test_getitem_single_file(self, naip: NAIP) -> None:
        x = naip[naip.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)
        assert len(naip.bands) == x['image'].shape[0]

    def test_getitem_separate_files(self, sentinel: Sentinel2) -> None:
        x = sentinel[sentinel.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['image'], torch.Tensor)
        assert len(sentinel.bands) == x['image'].shape[0]

    def test_reprojection(self, naip: NAIP) -> None:
        naip2 = NAIP(naip.paths, crs='EPSG:4326')
        assert naip.crs != naip2.crs
        assert not math.isclose(naip.res, naip2.res)

    @pytest.mark.parametrize('dtype', ['uint16', 'uint32'])
    def test_getitem_uint_dtype(self, dtype: str) -> None:
        root = os.path.join('tests', 'data', 'raster', dtype)
        ds = RasterDataset(root)
        x = ds[ds.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].dtype == torch.float32

    @pytest.mark.parametrize('dtype', [torch.float, torch.double])
    def test_resampling_float_dtype(self, dtype: torch.dtype) -> None:
        paths = os.path.join('tests', 'data', 'raster', 'uint16')
        ds = CustomRasterDataset(dtype, paths)
        x = ds[ds.bounds]
        assert x['image'].dtype == dtype
        assert ds.resampling == Resampling.bilinear

    @pytest.mark.parametrize('dtype', [torch.long, torch.bool])
    def test_resampling_int_dtype(self, dtype: torch.dtype) -> None:
        paths = os.path.join('tests', 'data', 'raster', 'uint16')
        ds = CustomRasterDataset(dtype, paths)
        x = ds[ds.bounds]
        assert x['image'].dtype == dtype
        assert ds.resampling == Resampling.nearest

    def test_invalid_query(self, sentinel: Sentinel2) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds: .*'
        ):
            sentinel[query]

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            RasterDataset(tmp_path)

    def test_no_all_bands(self) -> None:
        root = os.path.join('tests', 'data', 'sentinel2')
        bands = ['B04', 'B03', 'B02']
        transforms = nn.Identity()
        cache = True
        msg = (
            'CustomSentinelDataset is missing an `all_bands` attribute,'
            ' so `bands` cannot be specified.'
        )

        with pytest.raises(AssertionError, match=msg):
            CustomSentinelDataset(root, bands=bands, transforms=transforms, cache=cache)


class TestVectorDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> CustomVectorDataset:
        root = os.path.join('tests', 'data', 'vector')
        transforms = nn.Identity()
        return CustomVectorDataset(root, res=0.1, transforms=transforms)

    @pytest.fixture(scope='class')
    def multilabel(self) -> CustomVectorDataset:
        root = os.path.join('tests', 'data', 'vector')
        transforms = nn.Identity()
        return CustomVectorDataset(
            root, res=0.1, transforms=transforms, label_name='label_id'
        )

    def test_getitem(self, dataset: CustomVectorDataset) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(
            x['mask'].unique(),  # type: ignore[no-untyped-call]
            torch.tensor([0, 1], dtype=torch.uint8),
        )

    def test_time_index(self, dataset: CustomVectorDataset) -> None:
        assert dataset.index.bounds[4] > 0
        assert dataset.index.bounds[5] < sys.maxsize

    def test_getitem_multilabel(self, multilabel: CustomVectorDataset) -> None:
        x = multilabel[multilabel.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['crs'], CRS)
        assert isinstance(x['mask'], torch.Tensor)
        assert torch.equal(
            x['mask'].unique(),  # type: ignore[no-untyped-call]
            torch.tensor([0, 1, 2, 3], dtype=torch.uint8),
        )

    def test_empty_shapes(self, dataset: CustomVectorDataset) -> None:
        query = BoundingBox(1.1, 1.9, 1.1, 1.9, 0, sys.maxsize)
        x = dataset[query]
        assert torch.equal(x['mask'], torch.zeros(8, 8, dtype=torch.uint8))

    def test_invalid_query(self, dataset: CustomVectorDataset) -> None:
        query = BoundingBox(3, 3, 3, 3, 0, 0)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            VectorDataset(tmp_path)


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
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_4_epsg_4326'))
        transforms = nn.Identity()
        return IntersectionDataset(ds1, ds2, transforms=transforms)

    def test_getitem(self, dataset: IntersectionDataset) -> None:
        query = dataset.bounds
        sample = dataset[query]
        assert isinstance(sample['image'], torch.Tensor)

    def test_len(self, dataset: IntersectionDataset) -> None:
        assert len(dataset) == 1

    def test_str(self, dataset: IntersectionDataset) -> None:
        out = str(dataset)
        assert 'type: IntersectionDataset' in out
        assert 'bbox: BoundingBox' in out
        assert 'size: 1' in out

    def test_nongeo_dataset(self) -> None:
        ds1 = CustomNonGeoDataset()
        ds2 = CustomNonGeoDataset()
        with pytest.raises(
            ValueError, match='IntersectionDataset only supports GeoDatasets'
        ):
            IntersectionDataset(ds1, ds2)  # type: ignore[arg-type]

    def test_different_crs_12(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4326'))
        ds = IntersectionDataset(ds1, ds2)
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_12_3(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4326'))
        ds3 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_32631'))
        ds = (ds1 & ds2) & ds3
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds3) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_1_23(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4326'))
        ds3 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_32631'))
        ds = ds1 & (ds2 & ds3)
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds3) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_12(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_4_epsg_4087'))
        ds = IntersectionDataset(ds1, ds2)
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_12_3(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_4_epsg_4087'))
        ds3 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_8_epsg_4087'))
        ds = (ds1 & ds2) & ds3
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds3) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_1_23(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_4_epsg_4087'))
        ds3 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_8_epsg_4087'))
        ds = ds1 & (ds2 & ds3)
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds3) == len(ds) == 1
        assert isinstance(sample['image'], torch.Tensor)

    def test_point_dataset(self) -> None:
        ds1 = CustomGeoDataset(BoundingBox(0, 2, 2, 4, 4, 6))
        ds2 = CustomGeoDataset(BoundingBox(1, 1, 3, 3, 5, 5))
        ds = IntersectionDataset(ds1, ds2)
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == 1
        assert len(ds1) == len(ds2) == len(ds) == 1

    def test_no_overlap(self) -> None:
        ds1 = CustomGeoDataset(BoundingBox(0, 1, 2, 3, 4, 5))
        ds2 = CustomGeoDataset(BoundingBox(6, 7, 8, 9, 10, 11))
        msg = 'Datasets have no spatiotemporal intersection'
        with pytest.raises(RuntimeError, match=msg):
            IntersectionDataset(ds1, ds2)

    def test_grid_overlap(self) -> None:
        ds1 = CustomGeoDataset(BoundingBox(0, 1, 2, 3, 4, 5))
        ds2 = CustomGeoDataset(BoundingBox(1, 2, 3, 4, 5, 6))
        msg = 'Datasets have no spatiotemporal intersection'
        with pytest.raises(RuntimeError, match=msg):
            IntersectionDataset(ds1, ds2)

    def test_invalid_query(self, dataset: IntersectionDataset) -> None:
        query = BoundingBox(-1, -1, -1, -1, -1, -1)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]


class TestUnionDataset:
    @pytest.fixture(scope='class')
    def dataset(self) -> UnionDataset:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_4_epsg_4326'))
        transforms = nn.Identity()
        return UnionDataset(ds1, ds2, transforms=transforms)

    def test_getitem(self, dataset: UnionDataset) -> None:
        query = dataset.bounds
        sample = dataset[query]
        assert isinstance(sample['image'], torch.Tensor)

    def test_len(self, dataset: UnionDataset) -> None:
        assert len(dataset) == 2

    def test_str(self, dataset: UnionDataset) -> None:
        out = str(dataset)
        assert 'type: UnionDataset' in out
        assert 'bbox: BoundingBox' in out
        assert 'size: 2' in out

    def test_different_crs_12(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4326'))
        ds = UnionDataset(ds1, ds2)
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == 2
        assert len(ds1) == len(ds2) == 1
        assert len(ds) == 2
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_12_3(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4326'))
        ds3 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_32631'))
        ds = (ds1 | ds2) | ds3
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds3) == 1
        assert len(ds) == 3
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_crs_1_23(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4326'))
        ds3 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_32631'))
        ds = ds1 | (ds2 | ds3)
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds3) == 1
        assert len(ds) == 3
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_12(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_4_epsg_4087'))
        ds = UnionDataset(ds1, ds2)
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds.res == 2
        assert len(ds1) == len(ds2) == 1
        assert len(ds) == 2
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_12_3(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_4_epsg_4087'))
        ds3 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_8_epsg_4087'))
        ds = (ds1 | ds2) | ds3
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds3) == 1
        assert len(ds) == 3
        assert isinstance(sample['image'], torch.Tensor)

    def test_different_res_1_23(self) -> None:
        ds1 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_2_epsg_4087'))
        ds2 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_4_epsg_4087'))
        ds3 = RasterDataset(os.path.join('tests', 'data', 'raster', 'res_8_epsg_4087'))
        ds = ds1 | (ds2 | ds3)
        sample = ds[ds.bounds]
        assert ds1.crs == ds2.crs == ds3.crs == ds.crs == CRS.from_epsg(4087)
        assert ds1.res == ds2.res == ds3.res == ds.res == 2
        assert len(ds1) == len(ds2) == len(ds3) == 1
        assert len(ds) == 3
        assert isinstance(sample['image'], torch.Tensor)

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

    def test_invalid_query(self, dataset: UnionDataset) -> None:
        query = BoundingBox(-1, -1, -1, -1, -1, -1)
        with pytest.raises(
            IndexError, match='query: .* not found in index with bounds:'
        ):
            dataset[query]
