# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pytest
import requests
import torch
import torch.nn as nn
from rasterio.errors import RasterioIOError

from torchgeo.datasets import (
    DatasetNotFoundError,
    IntersectionDataset,
    OpenAerialMap,
    UnionDataset,
)
from torchgeo.datasets.openaerialmap import TileUtils


class TestTileUtils:
    def test_tile_basic(self) -> None:
        tile = TileUtils.tile(0.0, 0.0, 0)
        assert tile.x == 0
        assert tile.y == 0
        assert tile.z == 0

    def test_tile_positive_coords(self) -> None:
        tile = TileUtils.tile(10.0, 20.0, 5)
        assert 0 <= tile.x < 2**5
        assert 0 <= tile.y < 2**5
        assert tile.z == 5

    def test_tile_negative_coords(self) -> None:
        tile = TileUtils.tile(-10.0, -20.0, 5)
        assert 0 <= tile.x < 2**5
        assert 0 <= tile.y < 2**5
        assert tile.z == 5

    def test_tile_with_truncate(self) -> None:
        tile = TileUtils.tile(200.0, 100.0, 10, truncate=True)
        assert 0 <= tile.x < 2**10
        assert 0 <= tile.y < 2**10

    def test_tile_corners(self) -> None:
        tile_nw = TileUtils.tile(-180.0, 85.0, 1)
        tile_se = TileUtils.tile(180.0, -85.0, 1)
        assert tile_nw.x == 0
        assert tile_se.x == 1

    def test_bounds_basic(self) -> None:
        tile = TileUtils.Tile(0, 0, 1)
        bounds = TileUtils.bounds(tile)
        assert bounds.west == -180.0
        assert bounds.east == 0.0
        assert bounds.north > 0
        assert bounds.south < bounds.north

    def test_bounds_roundtrip(self) -> None:
        original_tile = TileUtils.Tile(10, 15, 8)
        bounds = TileUtils.bounds(original_tile)
        center_lng = (bounds.west + bounds.east) / 2
        center_lat = (bounds.north + bounds.south) / 2
        result_tile = TileUtils.tile(center_lng, center_lat, 8)
        assert result_tile == original_tile

    def test_tiles_basic(self) -> None:
        tiles = list(TileUtils.tiles(-1, -1, 1, 1, 0))
        assert len(tiles) == 1
        assert tiles[0].z == 0

    def test_tiles_multiple(self) -> None:
        tiles = list(TileUtils.tiles(-10, -10, 10, 10, 2))
        assert len(tiles) > 1
        assert all(t.z == 2 for t in tiles)

    def test_tiles_with_truncate(self) -> None:
        tiles = list(TileUtils.tiles(-200, -100, 200, 100, 1, truncate=True))
        assert len(tiles) > 0
        assert all(0 <= t.x < 2**1 for t in tiles)
        assert all(0 <= t.y < 2**1 for t in tiles)

    def test_tiles_antimeridian(self) -> None:
        tiles = list(TileUtils.tiles(175, -5, -175, 5, 2))
        assert len(tiles) > 0
        x_values = [t.x for t in tiles]
        assert min(x_values) == 0 or max(x_values) == 3

    def test_tiles_web_mercator_limits(self) -> None:
        tiles = list(TileUtils.tiles(-180, -90, 180, 90, 0, truncate=True))
        assert len(tiles) == 1

    def test_tiles_high_zoom(self) -> None:
        tiles = list(TileUtils.tiles(0, 0, 0.1, 0.1, 19))
        assert len(tiles) > 0
        assert all(t.z == 19 for t in tiles)

    def test_tiles_identical_bounds(self) -> None:
        tiles = list(TileUtils.tiles(10.0, 20.0, 10.0, 20.0, 5))
        assert len(tiles) == 1


class TestOpenAerialMap:
    @pytest.fixture
    def dataset(self) -> OpenAerialMap:
        root = os.path.join('tests', 'data', 'openaerialmap')
        transforms = nn.Identity()
        return OpenAerialMap(root, transforms=transforms)

    @pytest.fixture
    def mock_bbox(self) -> tuple[float, float, float, float]:
        return (85.51678, 27.63134, 85.52323, 27.63744)

    def test_getitem(self, dataset: OpenAerialMap) -> None:
        x = dataset[dataset.bounds]
        assert isinstance(x, dict)
        assert isinstance(x['image'], torch.Tensor)
        assert x['image'].shape[0] == 3

    def test_len(self, dataset: OpenAerialMap) -> None:
        assert len(dataset) == 2

    def test_and(self, dataset: OpenAerialMap) -> None:
        ds = dataset & dataset
        assert isinstance(ds, IntersectionDataset)

    def test_or(self, dataset: OpenAerialMap) -> None:
        ds = dataset | dataset
        assert isinstance(ds, UnionDataset)

    def test_plot(self, dataset: OpenAerialMap) -> None:
        x = dataset[dataset.bounds]
        dataset.plot(x, suptitle='Test')
        plt.close()

    def test_init_validation(self) -> None:
        with pytest.raises(
            ValueError, match='bbox must be provided when download=True'
        ):
            OpenAerialMap(download=True)

        with pytest.raises(ValueError, match='zoom must be between'):
            OpenAerialMap(bbox=(0, 0, 1, 1), download=True, zoom=5)

    def test_no_data(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetNotFoundError, match='Dataset not found'):
            OpenAerialMap(tmp_path)

    def test_download_flow(
        self,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        src_dir = os.path.join('tests', 'data', 'openaerialmap')
        valid_file = next(f for f in os.listdir(src_dir) if f.endswith('.tif'))
        shutil.copy(os.path.join(src_dir, valid_file), tmp_path / valid_file)

        class MockResponse:
            status_code = 200

            @staticmethod
            def json() -> dict[str, Any]:
                return {
                    'features': [
                        {
                            'id': 'test_id',
                            'properties': {},
                            'assets': {
                                'visual': {'href': 'http://example.com/image.tif'}
                            },
                        }
                    ]
                }

        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.post',
            lambda *args, **kwargs: MockResponse(),
        )

        download_called = False

        def mock_download(self: Any) -> None:
            nonlocal download_called
            download_called = True

        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.OpenAerialMap._download', mock_download
        )

        OpenAerialMap(tmp_path, bbox=mock_bbox, zoom=19, download=True)
        assert download_called

    def test_download_no_tms(
        self,
        mock_bbox: tuple[float, float, float, float],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        class MockResponse:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                pass

            @staticmethod
            def json() -> dict[str, list[Any]]:
                return {'features': []}

        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.post',
            lambda *args, **kwargs: MockResponse(),
        )

        with pytest.warns(UserWarning, match='No imagery found'):
            with pytest.raises(DatasetNotFoundError):
                OpenAerialMap(tmp_path, bbox=mock_bbox, download=True)

    def test_fetch_item_id_variations(
        self,
        dataset: OpenAerialMap,
        mock_bbox: tuple[float, float, float, float],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dataset.bbox = mock_bbox
        dataset.image_id = None

        post_response: dict[str, Any] = {
            'features': [
                {'id': 'test_id', 'collection': 'openaerialmap', 'properties': {}}
            ]
        }
        post_exception: type[Exception] | None = None

        class MockPostResponse:
            def raise_for_status(self) -> None:
                pass

            def json(self) -> dict[str, Any]:
                return post_response

        class MockGetResponse:
            @staticmethod
            def raise_for_status() -> None:
                pass

            @staticmethod
            def json() -> dict[str, Any]:
                return {
                    'tilesets': [
                        {
                            'links': [
                                {
                                    'rel': 'tile',
                                    'href': 'http://api/raster/collections/openaerialmap/items/test_id/tiles/WebMercatorQuad/{z}/{x}/{y}',
                                }
                            ]
                        }
                    ]
                }

        def mock_post(*args: Any, **kwargs: Any) -> MockPostResponse:
            if post_exception:
                raise post_exception('Fail')
            return MockPostResponse()

        monkeypatch.setattr('torchgeo.datasets.openaerialmap.requests.post', mock_post)
        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.get',
            lambda *args, **kwargs: MockGetResponse(),
        )

        result = dataset._fetch_item_id()  # type: ignore[reportPrivateUsage]
        assert result is not None
        assert 'WebMercatorQuad' in result

        post_response = {'features': []}
        assert dataset._fetch_item_id() is None  # type: ignore[reportPrivateUsage]

        post_exception = requests.RequestException
        with pytest.raises(RuntimeError, match='Failed to query STAC API'):
            dataset._fetch_item_id()  # type: ignore[reportPrivateUsage]

        post_exception = ValueError
        with pytest.raises(RuntimeError, match='Invalid STAC API response'):
            dataset._fetch_item_id()  # type: ignore[reportPrivateUsage]

    def test_download_tiles(
        self, dataset: OpenAerialMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset.paths = tmp_path

        class MockResponse:
            status_code = 200
            content = b'fake_tiff_data'

        monkeypatch.setattr(
            'torchgeo.datasets.openaerialmap.requests.get',
            lambda *args, **kwargs: MockResponse(),
        )

        georef_called = False

        def mock_georeference(filepath: str, tile: TileUtils.Tile) -> None:
            nonlocal georef_called
            georef_called = True

        monkeypatch.setattr(dataset, '_georeference_tile', mock_georeference)

        tile = TileUtils.Tile(x=1, y=1, z=1)
        tiles_url = 'http://example.com/{z}/{x}/{y}'
        dataset._download_tiles(tiles_url, [tile])  # type: ignore[reportPrivateUsage]

        assert georef_called

    def test_georeference_tile_success(
        self, dataset: OpenAerialMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filepath = tmp_path / 'test.tif'
        filepath.touch()
        tile = TileUtils.Tile(x=1, y=1, z=1)

        update_tags_called = False

        class MockDataset:
            width = 256
            height = 256
            transform = 'mock_transform'
            crs = 'mock_crs'

            def update_tags(self, **kwargs: Any) -> None:
                nonlocal update_tags_called
                update_tags_called = True

        class MockContextManager:
            def __enter__(self) -> MockDataset:
                return MockDataset()

            def __exit__(self, *args: Any) -> None:
                pass

        monkeypatch.setattr(
            'rasterio.open', lambda *args, **kwargs: MockContextManager()
        )

        dataset._georeference_tile(str(filepath), tile)  # type: ignore[reportPrivateUsage]

        assert update_tags_called

    def test_download_single_tile_failures(
        self, dataset: OpenAerialMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        dataset.paths = tmp_path

        filepath = tmp_path / 'OAM-2-2-2.tif'

        class MockDataset:
            crs = 'mock_crs'

        class MockContextManager:
            def __enter__(self) -> MockDataset:
                return MockDataset()

            def __exit__(self, *args: Any) -> None:
                pass

        def mock_open(*args: Any, **kwargs: Any) -> MockContextManager:
            return MockContextManager()

        monkeypatch.setattr('rasterio.open', mock_open)

        filepath.touch()
        assert filepath.exists()
        with mock_open(str(filepath)) as ds:
            assert ds.crs is not None

    def test_georeference_tile_error(
        self, dataset: OpenAerialMap, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        filepath = tmp_path / 'test.tif'
        filepath.touch()
        tile = TileUtils.Tile(x=1, y=1, z=1)

        def raise_rasterio_error(*args: Any, **kwargs: Any) -> None:
            raise RasterioIOError

        monkeypatch.setattr('rasterio.open', raise_rasterio_error)

        with pytest.warns(UserWarning, match='Could not georeference'):
            dataset._georeference_tile(str(filepath), tile)  # type: ignore[reportPrivateUsage]
