# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from pathlib import Path
from typing import Dict

import pytest
import torch
import torch.nn as nn
from _pytest.fixtures import SubRequest
from rasterio.crs import CRS
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    BoundingBox,
    GeoDataset,
    IntersectionDataset,
    Landsat8,
    RasterDataset,
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
        self.crs = crs
        self.res = res

    def __getitem__(self, query: BoundingBox) -> Dict[str, BoundingBox]:
        return {"index": query}


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

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoDataset()  # type: ignore[abstract]

    def test_and_vision(self, dataset: GeoDataset) -> None:
        ds2 = CustomVisionDataset()
        with pytest.raises(
            ValueError, match="IntersectionDataset only supports GeoDatasets"
        ):
            dataset & ds2  # type: ignore[operator]


class TestRasterDataset:
    @pytest.fixture(params=[True, False])
    def dataset(self, request: SubRequest) -> Landsat8:
        root = os.path.join("tests", "data", "landsat8")
        bands = ["B1", "B2", "B3", "B4", "B5", "B6", "B7"]
        crs = CRS.from_epsg(3005)
        transforms = nn.Identity()  # type: ignore[attr-defined]
        cache = request.param
        return Landsat8(root, bands=bands, crs=crs, transforms=transforms, cache=cache)

    def test_getitem(self, dataset: Landsat8) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x["crs"], CRS)
        assert isinstance(x["image"], torch.Tensor)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No RasterDataset data was found"):
            RasterDataset(str(tmp_path))


class TestVectorDataset:
    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No VectorDataset data was found"):
            VectorDataset(str(tmp_path))


class TestVisionDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> VisionDataset:
        return CustomVisionDataset()

    def test_getitem(self, dataset: VisionDataset) -> None:
        assert dataset[0] == {"index": 0}

    def test_len(self, dataset: VisionDataset) -> None:
        assert len(dataset) == 2

    def test_add_two(self) -> None:
        ds1 = CustomVisionDataset()
        ds2 = CustomVisionDataset()
        dataset = ds1 + ds2
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 4

    def test_add_three(self) -> None:
        ds1 = CustomVisionDataset()
        ds2 = CustomVisionDataset()
        ds3 = CustomVisionDataset()
        dataset = ds1 + ds2 + ds3
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 6

    def test_add_four(self) -> None:
        ds1 = CustomVisionDataset()
        ds2 = CustomVisionDataset()
        ds3 = CustomVisionDataset()
        ds4 = CustomVisionDataset()
        dataset = (ds1 + ds2) + (ds3 + ds4)
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 8

    def test_str(self, dataset: VisionDataset) -> None:
        assert "type: VisionDataset" in str(dataset)
        assert "size: 2" in str(dataset)

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            VisionDataset()  # type: ignore[abstract]


class TestVisionClassificationDataset:
    @pytest.fixture(scope="class")
    def dataset(self, root: str) -> VisionClassificationDataset:
        return VisionClassificationDataset(root)

    @pytest.fixture(scope="class")
    def root(self) -> str:
        root = os.path.join("tests", "data", "visionclassificationdataset")
        return root

    def test_getitem(self, dataset: VisionClassificationDataset) -> None:
        x = dataset[0]
        assert isinstance(x, dict)
        assert isinstance(x["image"], torch.Tensor)
        assert isinstance(x["label"], torch.Tensor)
        assert x["image"].shape[0] == 3

    def test_len(self, dataset: VisionClassificationDataset) -> None:
        assert len(dataset) == 2

    def test_add_two(self, root: str) -> None:
        ds1 = VisionClassificationDataset(root)
        ds2 = VisionClassificationDataset(root)
        dataset = ds1 + ds2
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 4

    def test_add_three(self, root: str) -> None:
        ds1 = VisionClassificationDataset(root)
        ds2 = VisionClassificationDataset(root)
        ds3 = VisionClassificationDataset(root)
        dataset = ds1 + ds2 + ds3
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 6

    def test_add_four(self, root: str) -> None:
        ds1 = VisionClassificationDataset(root)
        ds2 = VisionClassificationDataset(root)
        ds3 = VisionClassificationDataset(root)
        ds4 = VisionClassificationDataset(root)
        dataset = (ds1 + ds2) + (ds3 + ds4)
        assert isinstance(dataset, ConcatDataset)
        assert len(dataset) == 8

    def test_str(self, dataset: VisionClassificationDataset) -> None:
        assert "type: VisionDataset" in str(dataset)
        assert "size: 2" in str(dataset)


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
        assert len(dataset) == 2

    def test_str(self, dataset: IntersectionDataset) -> None:
        out = str(dataset)
        assert "type: IntersectionDataset" in out
        assert "bbox: BoundingBox" in out
        assert "size: 2" in out

    def test_vision_dataset(self) -> None:
        ds1 = CustomVisionDataset()
        ds2 = CustomVisionDataset()
        with pytest.raises(
            ValueError, match="IntersectionDataset only supports GeoDatasets"
        ):
            IntersectionDataset(ds1, ds2)  # type: ignore[arg-type]

    def test_different_crs(self) -> None:
        ds1 = CustomGeoDataset(crs=CRS.from_epsg(3005))
        ds2 = CustomGeoDataset(crs=CRS.from_epsg(32616))
        with pytest.raises(ValueError, match="Datasets must be in the same CRS"):
            IntersectionDataset(ds1, ds2)

    def test_different_res(self) -> None:
        ds1 = CustomGeoDataset(res=1)
        ds2 = CustomGeoDataset(res=2)
        with pytest.raises(ValueError, match="Datasets must have the same resolution"):
            IntersectionDataset(ds1, ds2)

    def test_no_overlap(self) -> None:
        ds1 = CustomGeoDataset(BoundingBox(0, 1, 2, 3, 4, 5))
        ds2 = CustomGeoDataset(BoundingBox(6, 7, 8, 9, 10, 11))
        with pytest.raises(ValueError, match="Datasets have no overlap"):
            IntersectionDataset(ds1, ds2)

    def test_invalid_query(self, dataset: IntersectionDataset) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
