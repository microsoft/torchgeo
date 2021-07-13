from typing import Any, Dict

import pytest
from rtree.index import Index, Property
from torch.utils.data import ConcatDataset

from torchgeo.datasets import BoundingBox, GeoDataset, VisionDataset, ZipDataset


class CustomGeoDataset(GeoDataset):
    def __init__(self) -> None:
        self.index = Index(properties=Property(dimension=3, interleaved=False))

    def __getitem__(self, query: BoundingBox) -> Dict[str, Any]:
        return {"index": query}


class CustomVisionDataset(VisionDataset):
    def __getitem__(self, index: int) -> Dict[str, Any]:
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

    def test_abstract(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            GeoDataset()  # type: ignore[abstract]

    def test_add_vision(self, dataset: GeoDataset) -> None:
        ds2 = CustomVisionDataset()
        with pytest.raises(
            AssertionError, match="ZipDataset only supports GeoDatasets"
        ):
            dataset + ds2  # type: ignore[operator]


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
        query = BoundingBox(0, 0, 0, 0, 0, 0)
        assert dataset[query] == {"index": query}

    def test_str(self, dataset: ZipDataset) -> None:
        assert "type: ZipDataset" in str(dataset)

    def test_invalid_dataset(self) -> None:
        ds1 = CustomVisionDataset()
        ds2 = CustomVisionDataset()
        with pytest.raises(
            AssertionError, match="ZipDataset only supports GeoDatasets"
        ):
            ZipDataset([ds1, ds2])  # type: ignore[list-item]
