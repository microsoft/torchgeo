# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import pickle
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from rasterio.crs import CRS
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    NAIP,
    BoundingBox,
    GeoDataset,
    IntersectionDataset,
    NonGeoClassificationDataset,
    NonGeoDataset,
    RasterDataset,
    Sentinel2,
    UnionDataset,
    VectorDataset,
    VisionClassificationDataset,
    VisionDataset,
)


class CustomGeoDataset(GeoDataset):
    def __init__(
        self,
        bounds: BoundingBox = BoundingBox(0, 1, 2, 3, 4, 5),
        crs: CRS = CRS.from_epsg(3005),
        res: float = 1,
    ) -> None:
        super().__init__()
        self.index.insert(0, tuple(bounds))
        self._crs = crs
        self.res = res

    def __getitem__(self, query: BoundingBox) -> Dict[str, BoundingBox]:
        return {"index": query}


class CustomVectorDataset(VectorDataset):
    filename_glob = "*.geojson"


class CustomNonGeoDataset(NonGeoDataset):
    def __getitem__(self, index: int) -> Dict[str, int]:
        return {"index": index}

    def __len__(self) -> int:
        return 2


class CustomVisionDataset(VisionDataset):
    def __getitem__(self, index: int) -> Dict[str, int]:
        return {"index": index}

    def __len__(self) -> int:
        return 2


class TestGeoDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> GeoDataset:
        return CustomGeoDataset()

    def test_getitem(self, dataset: GeoDataset) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        assert dataset[query] == {"index": query}

    def test_len(self, dataset: GeoDataset) -> None:
        assert len(dataset) == 1

    @pytest.mark.parametrize("crs", [CRS.from_epsg(3005), CRS.from_epsg(32616)])
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
        assert "type: GeoDataset" in out
        assert "bbox: BoundingBox" in out
        assert "size: 1" in out

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
            ValueError, match="IntersectionDataset only supports GeoDatasets"
        ):
            dataset & ds2  # type: ignore[operator]


class TestRasterDataset:
    @pytest.fixture(params=[True, False])
    def naip(self, request: SubRequest) -> NAIP:
        root = os.path.join("tests", "data", "naip")
        crs = CRS.from_epsg(3005)
        transforms = nn.Identity()
        cache = request.param
        return NAIP(root, crs=crs, transforms=transforms, cache=cache)

    @pytest.fixture(params=[True, False])
    def sentinel(self, request: SubRequest) -> Sentinel2:
        root = os.path.join("tests", "data", "sentinel2")
        bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B09", "B11"]
        transforms = nn.Identity()
        cache = request.param
        return Sentinel2(root, bands=bands, transforms=transforms, cache=cache)

    @pytest.fixture()
    def custom_dtype_ds(self) -> RasterDataset:
        root = os.path.join("tests", "data", "raster")
        return RasterDataset(root)

    def test_getitem_single_file(self, naip: NAIP) -> None:
        x = naip[naip.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_getitem_separate_files(self, sentinel: Sentinel2) -> None:
        x = sentinel[sentinel.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_getitem_uint_dtype(self, custom_dtype_ds: RasterDataset) -> None:
        x = custom_dtype_ds[custom_dtype_ds.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert x["image"].dtype == torch.int64

    def test_invalid_query(self, sentinel: Sentinel2) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds: .*"
        ):
            sentinel[query]

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No RasterDataset data was found"):
            RasterDataset(str(tmp_path))


class TestVectorDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> CustomVectorDataset:
        root = os.path.join("tests", "data", "vector")
        transforms = nn.Identity()
        return CustomVectorDataset(root, res=0.1, transforms=transforms)

    def test_getitem(self, dataset: CustomVectorDataset) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["mask"], torch.Tensor)

    def test_empty_shapes(self, dataset: CustomVectorDataset) -> None:
        query = BoundingBox(1.1, 1.9, 1.1, 1.9, 0, 0)
        x = dataset[query]
        assert torch.equal(x["mask"], torch.zeros(8, 8, dtype=torch.uint8))

    def test_invalid_query(self, dataset: CustomVectorDataset) -> None:
        query = BoundingBox(3, 3, 3, 3, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No VectorDataset data was found"):
            VectorDataset(str(tmp_path))


class TestNonGeoDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> NonGeoDataset:
        return CustomNonGeoDataset()

    def test_getitem(self, dataset: NonGeoDataset) -> None:
        assert dataset[0] == {"index": 0}

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
        assert "type: NonGeoDataset" in str(dataset)
        assert "size: 2" in str(dataset)

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            NonGeoDataset()  # type: ignore[abstract]


class TestVisionDataset:
    def test_deprecation(self) -> None:
        match = "VisionDataset is deprecated, use NonGeoDataset instead."
        with pytest.warns(DeprecationWarning, match=match):
            CustomVisionDataset()


class TestNonGeoClassificationDataset:
    @pytest.fixture(scope="class")
    def dataset(self, root: str) -> NonGeoClassificationDataset:
        transforms = nn.Identity()
        return NonGeoClassificationDataset(root, transforms=transforms)

    @pytest.fixture(scope="class")
    def root(self) -> str:
        root = os.path.join("tests", "data", "nongeoclassification")
        return root

    def test_getitem(self, dataset: NonGeoClassificationDataset) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3

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
        assert "type: NonGeoDataset" in str(dataset)
        assert "size: 2" in str(dataset)


class TestVisionClassificationDataset:
    def test_deprecation(self) -> None:
        root = os.path.join("tests", "data", "nongeoclassification")
        match = "VisionClassificationDataset is deprecated, "
        match += "use NonGeoClassificationDataset instead."
        with pytest.warns(DeprecationWarning, match=match):
            VisionClassificationDataset(root)


class TestIntersectionDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> IntersectionDataset:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        return ds1 & ds2

    def test_getitem(self, dataset: IntersectionDataset) -> None:
        query = BoundingBox(0, 1, 2, 3, 4, 5)
        assert dataset[query] == {"index": query}

    def test_len(self, dataset: IntersectionDataset) -> None:
        assert len(dataset) == 1

    def test_str(self, dataset: IntersectionDataset) -> None:
        out = str(dataset)
        assert "type: IntersectionDataset" in out
        assert "bbox: BoundingBox" in out
        assert "size: 1" in out

    def test_nongeo_dataset(self) -> None:
        ds1 = CustomNonGeoDataset()
        ds2 = CustomNonGeoDataset()
        with pytest.raises(
            ValueError, match="IntersectionDataset only supports GeoDatasets"
        ):
            IntersectionDataset(ds1, ds2)  # type: ignore[arg-type]

    def test_different_crs(self) -> None:
        ds1 = CustomGeoDataset(crs=CRS.from_epsg(3005))
        ds2 = CustomGeoDataset(crs=CRS.from_epsg(32616))
        ds = IntersectionDataset(ds1, ds2)
        assert len(ds) == 0

    def test_different_res(self) -> None:
        ds1 = CustomGeoDataset(res=1)
        ds2 = CustomGeoDataset(res=2)
        ds = IntersectionDataset(ds1, ds2)
        assert len(ds) == 1

    def test_no_overlap(self) -> None:
        ds1 = CustomGeoDataset(BoundingBox(0, 1, 2, 3, 4, 5))
        ds2 = CustomGeoDataset(BoundingBox(6, 7, 8, 9, 10, 11))
        ds = IntersectionDataset(ds1, ds2)
        assert len(ds) == 0

    def test_invalid_query(self, dataset: IntersectionDataset) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]


class TestUnionDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> UnionDataset:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        return ds1 | ds2

    def test_getitem(self, dataset: UnionDataset) -> None:
        query = BoundingBox(0, 1, 2, 3, 4, 5)
        assert dataset[query] == {"index": query}

    def test_len(self, dataset: UnionDataset) -> None:
        assert len(dataset) == 2

    def test_str(self, dataset: UnionDataset) -> None:
        out = str(dataset)
        assert "type: UnionDataset" in out
        assert "bbox: BoundingBox" in out
        assert "size: 2" in out

    def test_nongeo_dataset(self) -> None:
        ds1 = CustomNonGeoDataset()
        ds2 = CustomNonGeoDataset()
        with pytest.raises(ValueError, match="UnionDataset only supports GeoDatasets"):
            UnionDataset(ds1, ds2)  # type: ignore[arg-type]

    def test_different_crs(self) -> None:
        ds1 = CustomGeoDataset(crs=CRS.from_epsg(3005))
        ds2 = CustomGeoDataset(crs=CRS.from_epsg(32616))
        ds = UnionDataset(ds1, ds2)
        assert len(ds) == 2

    def test_different_res(self) -> None:
        ds1 = CustomGeoDataset(res=1)
        ds2 = CustomGeoDataset(res=2)
        ds = UnionDataset(ds1, ds2)
        assert len(ds) == 2

    def test_no_overlap(self) -> None:
        ds1 = CustomGeoDataset(BoundingBox(0, 1, 2, 3, 4, 5))
        ds2 = CustomGeoDataset(BoundingBox(6, 7, 8, 9, 10, 11))
        ds = UnionDataset(ds1, ds2)
        assert len(ds) == 2

    def test_invalid_query(self, dataset: UnionDataset) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
