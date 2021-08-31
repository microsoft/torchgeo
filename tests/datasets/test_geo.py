# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from pathlib import Path
from typing import Dict

import pytest
from rasterio.crs import CRS
from torch.utils.data import ConcatDataset

from torchgeo.datasets import (
    BoundingBox,
    GeoDataset,
    RasterDataset,
    VectorDataset,
    VisionDataset,
    ZipDataset,
)


class CustomGeoDataset(GeoDataset):
    def __init__(
        self,
        bounds: BoundingBox = BoundingBox(0, 1, 2, 3, 4, 5),
        crs: CRS = CRS.from_epsg(3005),
        res: float = 1,
    ) -> None:
        super().__init__()
        self.index.insert(0, bounds)
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

    def test_add_two(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        dataset = ds1 + ds2
        assert isinstance(dataset, ZipDataset)

    def test_add_three(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        ds3 = CustomGeoDataset()
        dataset = ds1 + ds2 + ds3
        assert isinstance(dataset, ZipDataset)

    def test_add_four(self) -> None:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        ds3 = CustomGeoDataset()
        ds4 = CustomGeoDataset()
        dataset = (ds1 + ds2) + (ds3 + ds4)
        assert isinstance(dataset, ZipDataset)

    def test_str(self, dataset: GeoDataset) -> None:
        assert "type: GeoDataset" in str(dataset)
        assert "bbox: BoundingBox" in str(dataset)

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoDataset()  # type: ignore[abstract]

    def test_add_vision(self, dataset: GeoDataset) -> None:
        ds2 = CustomVisionDataset()
        with pytest.raises(ValueError, match="ZipDataset only supports GeoDatasets"):
            dataset + ds2  # type: ignore[operator]


class TestRasterDataset:
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


class TestZipDataset:
    @pytest.fixture(scope="class")
    def dataset(self) -> ZipDataset:
        ds1 = CustomGeoDataset()
        ds2 = CustomGeoDataset()
        return ZipDataset([ds1, ds2])

    def test_getitem(self, dataset: ZipDataset) -> None:
        query = BoundingBox(0, 1, 2, 3, 4, 5)
        assert dataset[query] == {"index": query}

    def test_str(self, dataset: ZipDataset) -> None:
        assert "type: ZipDataset" in str(dataset)
        assert "bbox: BoundingBox" in str(dataset)

    def test_vision_dataset(self) -> None:
        ds1 = CustomVisionDataset()
        ds2 = CustomVisionDataset()
        with pytest.raises(ValueError, match="ZipDataset only supports GeoDatasets"):
            ZipDataset([ds1, ds2])  # type: ignore[list-item]

    def test_different_crs(self) -> None:
        ds1 = CustomGeoDataset(crs=CRS.from_epsg(3005))
        ds2 = CustomGeoDataset(crs=CRS.from_epsg(32616))
        with pytest.raises(ValueError, match="Datasets must be in the same CRS"):
            ZipDataset([ds1, ds2])

    def test_different_res(self) -> None:
        ds1 = CustomGeoDataset(res=1)
        ds2 = CustomGeoDataset(res=2)
        with pytest.raises(ValueError, match="Datasets must have the same resolution"):
            ZipDataset([ds1, ds2])

    def test_no_overlap(self) -> None:
        ds1 = CustomGeoDataset(BoundingBox(0, 1, 2, 3, 4, 5))
        ds2 = CustomGeoDataset(BoundingBox(6, 7, 8, 9, 10, 11))
        with pytest.raises(ValueError, match="Datasets have no overlap"):
            ZipDataset([ds1, ds2])

    def test_invalid_query(self, dataset: ZipDataset) -> None:
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        with pytest.raises(
            IndexError, match="query: .* not found in index with bounds:"
        ):
            dataset[query]
