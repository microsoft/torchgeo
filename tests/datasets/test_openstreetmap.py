# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import json
import os
import pathlib
import time
import warnings
from typing import Any, NoReturn

import geopandas as gpd
import pandas as pd
import pytest
import torch
from pytest import MonkeyPatch
from shapely.geometry import Point

from torchgeo.datasets import DatasetNotFoundError, OpenStreetMap


class TestOpenStreetMap:
    @pytest.fixture
    def dataset(self) -> OpenStreetMap:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        classes = [{'name': 'building', 'selector': [{'building': '*'}]}]
        return OpenStreetMap(bbox=bbox, classes=classes, paths=root, download=False)

    @pytest.fixture
    def mock_download_and_integrity(self, monkeypatch: MonkeyPatch) -> None:
        """Fixture to mock download and integrity check for most tests."""
        monkeypatch.setattr(OpenStreetMap, '_download_data', lambda _: None)
        monkeypatch.setattr(OpenStreetMap, '_check_integrity', lambda _: True)

    @pytest.fixture
    def common_test_params(self) -> dict[str, Any]:
        """Common test parameters used across multiple tests."""
        return {
            'root': os.path.join('tests', 'data', 'openstreetmap'),
            'bbox': (2.3520, 48.8565, 2.3525, 48.8570),
            'classes': [{'name': 'building', 'selector': [{'building': '*'}]}],
        }

    @pytest.fixture
    def multi_channel_params(self) -> dict[str, Any]:
        """Parameters for multi-class tests to reduce file duplication."""
        return {
            'root': os.path.join('tests', 'data', 'openstreetmap'),
            'bbox': (2.3520, 48.8565, 2.3525, 48.8570),
            'classes': [
                {'name': 'building', 'selector': [{'building': '*'}]},
                {'name': 'amenity', 'selector': [{'amenity': '*'}]},
                {'name': 'highway', 'selector': [{'highway': '*'}]},
            ],
        }

    def test_init_no_download(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        # Use a different bbox that won't have test data
        bbox = (0.0, 0.0, 0.001, 0.001)
        classes = [{'name': 'buildings', 'selector': [{'building': '*'}]}]
        with pytest.raises(DatasetNotFoundError):
            OpenStreetMap(bbox=bbox, classes=classes, paths=root, download=False)

    def test_init_with_download(
        self, mock_download_and_integrity: None, common_test_params: dict[str, Any]
    ) -> None:
        dataset = OpenStreetMap(
            bbox=common_test_params['bbox'],
            classes=common_test_params['classes'],
            paths=common_test_params['root'],
            download=True,
        )
        assert dataset.bbox == common_test_params['bbox']
        assert dataset.classes == common_test_params['classes']

    def test_custom_query(self, mock_download_and_integrity: None) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Test custom selector combinations in classes
        classes = [
            {
                'name': 'mixed_features',
                'selector': [{'building': '*'}, {'leisure': 'park'}],
            }
        ]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=classes)
        assert dataset.classes == classes

    def test_build_overpass_query(self, mock_download_and_integrity: None) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        classes = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, classes=classes, paths=root)
        query = dataset._build_overpass_query()
        assert 'wr["building"]' in query
        assert (
            '48.8565,2.352,48.857,2.3525' in query
        )  # bbox format: south,west,north,east

    def test_get_data_filename(self, mock_download_and_integrity: None) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels)
        filename = dataset._get_data_filename()
        assert filename.suffix == '.geojson'
        assert 'osm_features' in filename.name

    def test_getitem(
        self, mock_download_and_integrity: None, monkeypatch: MonkeyPatch
    ) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Mock GeoDataFrame with building and label
        mock_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(2.3523, 48.8567)], 'building': ['yes'], 'label': [1]}
        )

        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels)
        # Use dataset bounds for querying
        sample = dataset[dataset.bounds]

        # VectorDataset in semantic segmentation mode returns mask
        assert 'mask' in sample

    @pytest.mark.parametrize(
        'plot_kwargs',
        [
            {},  # default plot
            {'suptitle': 'Test Title'},  # with suptitle
            {'show_titles': False},  # without titles
        ],
    )
    def test_plot_with_empty_data(
        self,
        mock_download_and_integrity: None,
        common_test_params: dict[str, Any],
        monkeypatch: MonkeyPatch,
        plot_kwargs: dict[str, Any],
    ) -> None:
        """Test plot method with empty data and different parameters."""
        mock_gdf = gpd.GeoDataFrame({'geometry': []})
        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        dataset = OpenStreetMap(
            bbox=common_test_params['bbox'],
            paths=common_test_params['root'],
            classes=common_test_params['classes'],
        )
        sample = {'mask': torch.zeros((10, 10))}  # Typical sample format
        fig = dataset.plot(sample, **plot_kwargs)

        assert fig is not None

    @pytest.mark.parametrize(
        'paths_input,expected_root_name',
        [
            ('tests/data/openstreetmap', 'openstreetmap'),  # string path
            (['tests/data/openstreetmap'], 'openstreetmap'),  # list path
            (pathlib.Path('tests/data/openstreetmap'), 'openstreetmap'),  # Path object
        ],
    )
    def test_paths_parameter_variations(
        self,
        mock_download_and_integrity: None,
        common_test_params: dict[str, Any],
        paths_input: Any,
        expected_root_name: str,
    ) -> None:
        """Test different types and formats for paths parameter."""
        dataset = OpenStreetMap(
            bbox=common_test_params['bbox'],
            paths=paths_input,
            classes=common_test_params['classes'],
        )
        assert dataset.root.name == expected_root_name
        assert isinstance(dataset.root, pathlib.Path)

    def test_check_integrity(self, monkeypatch: MonkeyPatch) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        monkeypatch.setattr(OpenStreetMap, '_download_data', lambda _: None)

        # Test _check_integrity when file doesn't exist
        dataset = OpenStreetMap.__new__(
            OpenStreetMap
        )  # Create instance without __init__
        dataset.bbox = bbox
        dataset.classes = [{'name': 'nonexistent', 'selector': [{'nonexistent': '*'}]}]
        dataset.root = pathlib.Path(root)
        assert not dataset._check_integrity()

    def test_build_overpass_query_multiple_selectors(
        self, mock_download_and_integrity: None
    ) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Test query with multiple selectors in one channel
        channels = [
            {'name': 'mixed', 'selector': [{'building': '*'}, {'leisure': 'park'}]}
        ]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels)
        query = dataset._build_overpass_query()
        assert 'wr["building"]' in query
        assert 'wr["leisure"="park"]' in query
        assert '48.8565,2.352,48.857,2.3525' in query

    def test_build_overpass_query_list_values(
        self, common_test_params: dict[str, Any]
    ) -> None:
        """Test Overpass query generation with list values in selectors."""
        # Create a dataset with list values in selector
        channels = [
            {'name': 'roads', 'selector': [{'highway': ['primary', 'secondary']}]}
        ]

        # Create a temporary instance to test the query generation
        dataset = OpenStreetMap.__new__(OpenStreetMap)
        dataset.bbox = common_test_params['bbox']
        dataset.classes = channels

        query = dataset._build_overpass_query()

        # Should generate regex for multiple values
        assert 'wr["highway"~"^(primary|secondary)$"]' in query
        assert '48.8565,2.352,48.857,2.3525' in query

    def test_rate_limiting(
        self, mock_download_and_integrity: None, monkeypatch: MonkeyPatch
    ) -> None:
        """Test rate limiting functionality."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Reset class variable for test
        monkeypatch.setattr(OpenStreetMap, '_last_request_time', 0.0)

        start_time = time.time()
        dataset._rate_limit()
        dataset._rate_limit()
        end_time = time.time()

        # Should take at least the minimum interval (with small tolerance for timing)
        assert end_time - start_time >= dataset._min_request_interval * 0.9

    def test_parse_overpass_response(self, mock_download_and_integrity: None) -> None:
        """Test parsing of Overpass API response."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Mock OSM API response with different element types
        mock_osm_response = {
            'elements': [
                {
                    'type': 'node',
                    'id': 123,
                    'lat': 48.8566,
                    'lon': 2.3523,
                    'tags': {'amenity': 'restaurant', 'name': 'Test Restaurant'},
                },
                {
                    'type': 'way',
                    'id': 456,
                    'geometry': [
                        {'lat': 48.8565, 'lon': 2.3520},
                        {'lat': 48.8570, 'lon': 2.3525},
                    ],
                    'tags': {'highway': 'primary'},
                },
            ]
        }

        # Test parsing response (covers lines 262-278)
        gdf = dataset._parse_overpass_response(mock_osm_response)

        assert len(gdf) == 2
        assert 'osm_id' in gdf.columns
        assert 'osm_type' in gdf.columns
        assert gdf.iloc[0]['osm_id'] == 123
        assert gdf.iloc[0]['osm_type'] == 'node'
        assert gdf.iloc[0]['amenity'] == 'restaurant'

    def test_parse_overpass_response_empty(
        self, mock_download_and_integrity: None
    ) -> None:
        """Test parsing empty Overpass API response."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Empty response
        empty_response: dict[str, Any] = {'elements': []}
        gdf = dataset._parse_overpass_response(empty_response)

        assert len(gdf) == 0
        assert gdf.crs == 'EPSG:4326'

    @pytest.mark.parametrize(
        'element,expected_geom_type,extra_checks',
        [
            # Valid elements
            (
                {'type': 'node', 'lat': 48.8566, 'lon': 2.3523},
                'Point',
                lambda g: g.x == 2.3523 and g.y == 48.8566,
            ),
            (
                {
                    'type': 'way',
                    'geometry': [
                        {'lat': 48.8565, 'lon': 2.3520},
                        {'lat': 48.8570, 'lon': 2.3525},
                    ],
                },
                'LineString',
                lambda g: True,
            ),
            (
                {
                    'type': 'way',
                    'geometry': [
                        {'lat': 48.8566, 'lon': 2.3522},
                        {'lat': 48.8566, 'lon': 2.3524},
                        {'lat': 48.8568, 'lon': 2.3524},
                        {'lat': 48.8568, 'lon': 2.3522},
                        {'lat': 48.8566, 'lon': 2.3522},  # Closed polygon
                    ],
                },
                'Polygon',
                lambda g: True,
            ),
            # Invalid elements should return None
            ({'type': 'node'}, None, lambda g: True),  # Missing lat/lon
            ({'type': 'way'}, None, lambda g: True),  # Missing geometry
            ({'type': 'unknown'}, None, lambda g: True),  # Unknown type
            (
                {'type': 'way', 'geometry': [{'lat': 48.8565, 'lon': 2.3520}]},
                None,
                lambda g: True,
            ),  # Single point way
        ],
    )
    def test_element_to_geometry(
        self,
        mock_download_and_integrity: None,
        common_test_params: dict[str, Any],
        element: dict[str, Any],
        expected_geom_type: str | None,
        extra_checks: Any,
    ) -> None:
        """Test converting OSM elements to geometries."""
        dataset = OpenStreetMap(
            bbox=common_test_params['bbox'],
            paths=common_test_params['root'],
            classes=common_test_params['classes'],
            download=False,
        )

        geom = dataset._element_to_geometry(element)

        if expected_geom_type is None:
            assert geom is None
        else:
            assert geom is not None
            assert geom.geom_type == expected_geom_type
            assert extra_checks(geom)

    def test_getitem_empty_data(self, monkeypatch: MonkeyPatch) -> None:
        """Test __getitem__ when no data is found in query area."""

        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Mock empty GeoDataFrame
        empty_gdf = gpd.GeoDataFrame({'geometry': []})

        monkeypatch.setattr(OpenStreetMap, '_download_data', lambda _: None)
        monkeypatch.setattr(OpenStreetMap, '_check_integrity', lambda _: True)
        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: empty_gdf)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)
        sample = dataset[dataset.bounds]

        # Should create mask for semantic segmentation mode
        assert 'mask' in sample

    def test_plot_without_bounds_key(self, monkeypatch: MonkeyPatch) -> None:
        """Test plot method with a proper sample from dataset."""

        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Mock GeoDataFrame with actual data
        mock_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(2.3523, 48.8567)], 'label': [1]}
        )

        monkeypatch.setattr(OpenStreetMap, '_download_data', lambda _: None)
        monkeypatch.setattr(OpenStreetMap, '_check_integrity', lambda _: True)
        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels)

        # Get proper sample from dataset
        query = dataset.bounds
        sample = dataset[query]
        fig = dataset.plot(sample)

        assert fig is not None

    def test_plot_with_actual_data(self, monkeypatch: MonkeyPatch) -> None:
        """Test plot method with actual geometry data."""

        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Mock GeoDataFrame with actual data and labels
        mock_gdf = gpd.GeoDataFrame(
            {
                'geometry': [Point(2.3523, 48.8567)],
                'amenity': ['restaurant'],
                'label': [1],
            }
        )

        monkeypatch.setattr(OpenStreetMap, '_download_data', lambda _: None)
        monkeypatch.setattr(OpenStreetMap, '_check_integrity', lambda _: True)
        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels)

        # Get proper sample from dataset
        query = dataset.bounds
        sample = dataset[query]

        fig = dataset.plot(sample)
        assert fig is not None

    def test_download_data_success(self, monkeypatch: MonkeyPatch) -> None:
        """Test successful data download."""

        # Create proper mock context manager like in old tests
        mock_response_data = {
            'elements': [
                {
                    'type': 'node',
                    'id': 123,
                    'lat': 48.8566,
                    'lon': 2.3523,
                    'tags': {'amenity': 'restaurant'},
                }
            ]
        }

        class MockResponse:
            def read(self) -> bytes:
                return json.dumps(mock_response_data).encode('utf-8')

        class MockUrlOpen:
            def __enter__(self) -> MockResponse:
                return MockResponse()

            def __exit__(self, *args: Any) -> None:
                return None

        monkeypatch.setattr(
            'torchgeo.datasets.openstreetmap.urlopen', lambda *_, **__: MockUrlOpen()
        )

        # Use unique bbox to avoid cache conflicts
        unique_offset = time.time() % 1000 / 10000
        bbox = (
            2.4 + unique_offset,
            48.9 + unique_offset,
            2.401 + unique_offset,
            48.901 + unique_offset,
        )
        root = os.path.join('tests', 'data', 'openstreetmap')

        # Create dataset which should trigger download (covers lines 216-239)
        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=True)

        # Check data file was created
        data_file = dataset._get_data_filename()
        assert data_file.exists()
        os.remove(data_file)

    def test_download_empty_response_raises_error(
        self, monkeypatch: MonkeyPatch, tmp_path: pathlib.Path
    ) -> None:
        """Test download method raises ValueError for empty API response."""
        # Mock empty API response
        mock_response_data: dict[str, list[Any]] = {'elements': []}

        class MockResponse:
            def read(self) -> bytes:
                return json.dumps(mock_response_data).encode('utf-8')

        class MockUrlOpen:
            def __enter__(self) -> MockResponse:
                return MockResponse()

            def __exit__(self, *_: Any) -> None:
                return None

        monkeypatch.setattr(
            'torchgeo.datasets.openstreetmap.urlopen', lambda *_, **__: MockUrlOpen()
        )

        # Test both direct call and through constructor
        # Direct call to _download_data
        dataset = OpenStreetMap.__new__(OpenStreetMap)
        dataset.bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        dataset.classes = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset.root = tmp_path

        with pytest.raises(ValueError, match='No features found in the specified area'):
            dataset._download_data()

        # Through constructor with unique bbox
        unique_offset = time.time() % 1000 / 10000
        bbox = (
            2.7 + unique_offset,
            48.97 + unique_offset,
            2.701 + unique_offset,
            48.971 + unique_offset,
        )
        root = os.path.join('tests', 'data', 'openstreetmap')
        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]

        with pytest.raises(ValueError, match='No features found in the specified area'):
            OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=True)

    def test_download_data_all_endpoints_fail(self, monkeypatch: MonkeyPatch) -> None:
        """Test download failure when all endpoints fail."""

        # Make all requests fail like in old tests
        def mock_urlopen_fail(*_: Any) -> NoReturn:
            raise Exception('Connection failed')

        monkeypatch.setattr(
            'torchgeo.datasets.openstreetmap.urlopen', mock_urlopen_fail
        )

        # Use unique bbox that doesn't have cached data
        unique_offset = time.time() % 1000 / 10000
        bbox = (
            2.6 + unique_offset,
            48.96 + unique_offset,
            2.601 + unique_offset,
            48.961 + unique_offset,
        )
        root = os.path.join('tests', 'data', 'openstreetmap')

        # Should raise RuntimeError when all endpoints fail (covers line 251)
        monkeypatch.setattr(OpenStreetMap, '_rate_limit', lambda _: None)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        with pytest.raises(RuntimeError, match='All Overpass API endpoints failed'):
            OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=True)

    def test_download_data_file_exists(self, monkeypatch: MonkeyPatch) -> None:
        """Test _download_data when file already exists."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)  # Use existing test data

        urlopen_called = False

        def mock_urlopen(*_: Any) -> NoReturn:
            nonlocal urlopen_called
            urlopen_called = True
            raise Exception('Should not be called')

        # Use existing dataset that has cached data (covers lines 212-213)
        monkeypatch.setattr('urllib.request.urlopen', mock_urlopen)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)
        dataset._download_data()  # Should return early without network calls
        assert not urlopen_called

    # New tests for channels functionality and pre-computed labels

    @pytest.mark.parametrize(
        'channels,expected_error,error_pattern',
        [
            # Valid case - should not raise
            ([{'name': 'building', 'selector': [{'building': '*'}]}], None, None),
            # Invalid cases
            ([], ValueError, 'classes must be a non-empty list'),
            ('invalid', ValueError, 'classes must be a non-empty list'),
            (['invalid'], ValueError, 'Class 0 must be a dictionary'),
            (
                [{'name': 'test'}],
                ValueError,
                'Class 0 must have "name" and "selector" keys',
            ),
            (
                [{'name': 'test', 'selector': 'invalid'}],
                ValueError,
                'Class 0 selector must be a list',
            ),
            (
                [{'name': 'test', 'selector': ['invalid']}],
                ValueError,
                'Class 0 selector 0 must be a dictionary',
            ),
        ],
    )
    def test_validate_classes(
        self,
        common_test_params: dict[str, Any],
        channels: Any,
        expected_error: Any,
        error_pattern: str | None,
    ) -> None:
        """Test channel validation with various valid and invalid inputs."""
        if expected_error is None:
            # Valid case - should not raise
            dataset = OpenStreetMap(
                bbox=common_test_params['bbox'],
                paths=common_test_params['root'],
                classes=channels,
                download=False,
            )
            assert dataset.classes == channels
        else:
            # Invalid case - should raise the expected error
            with pytest.raises(expected_error, match=error_pattern):
                OpenStreetMap(
                    bbox=common_test_params['bbox'],
                    paths=common_test_params['root'],
                    classes=channels,
                    download=False,
                )

    def test_build_overpass_query_classes(
        self, multi_channel_params: dict[str, Any]
    ) -> None:
        """Test Overpass query building with channels."""
        dataset = OpenStreetMap(
            bbox=multi_channel_params['bbox'],
            paths=multi_channel_params['root'],
            classes=multi_channel_params['classes'],
            download=False,
        )
        query = dataset._build_overpass_query()

        assert 'wr["building"]' in query
        assert 'wr["amenity"]' in query
        assert 'wr["highway"]' in query
        assert '48.8565,2.352,48.857,2.3525' in query

    def test_get_data_filename_classes(
        self, common_test_params: dict[str, Any]
    ) -> None:
        """Test filename generation for channels-based datasets."""
        dataset = OpenStreetMap(
            bbox=common_test_params['bbox'],
            paths=common_test_params['root'],
            classes=common_test_params['classes'],
            download=False,
        )
        filename = dataset._get_data_filename()

        assert filename.suffix == '.geojson'
        assert 'osm_features_' in filename.name

    def test_get_class_label(self, multi_channel_params: dict[str, Any]) -> None:
        """Test label computation based on channels."""
        dataset = OpenStreetMap(
            bbox=multi_channel_params['bbox'],
            paths=multi_channel_params['root'],
            classes=multi_channel_params['classes'],
            download=False,
        )

        # Test building feature -> label 1 (first channel)
        building_feature = {'properties': {'building': 'yes'}}
        assert dataset._get_class_label(building_feature) == 1

        # Test amenity feature -> label 2 (second channel)
        amenity_feature = {'properties': {'amenity': 'restaurant'}}
        assert dataset._get_class_label(amenity_feature) == 2

        # Test highway feature -> label 3 (third channel)
        highway_feature = {'properties': {'highway': 'primary'}}
        assert dataset._get_class_label(highway_feature) == 3

        # Test no match -> label 0
        other_feature = {'properties': {'shop': 'bakery'}}
        assert dataset._get_class_label(other_feature) == 0

    def test_feature_matches_selector(self) -> None:
        """Test feature matching logic."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Test wildcard match
        assert dataset._feature_matches_selector({'building': 'yes'}, {'building': '*'})
        assert not dataset._feature_matches_selector(
            {'shop': 'bakery'}, {'building': '*'}
        )

        # Test exact match
        assert dataset._feature_matches_selector(
            {'leisure': 'park'}, {'leisure': 'park'}
        )
        assert not dataset._feature_matches_selector(
            {'leisure': 'garden'}, {'leisure': 'park'}
        )

        # Test list match
        assert dataset._feature_matches_selector(
            {'highway': 'primary'}, {'highway': ['primary', 'secondary']}
        )
        assert not dataset._feature_matches_selector(
            {'highway': 'residential'}, {'highway': ['primary', 'secondary']}
        )

        # Test missing property
        assert not dataset._feature_matches_selector({}, {'building': '*'})

        # Test None value (line 457)
        assert not dataset._feature_matches_selector(
            {'building': None}, {'building': '*'}
        )

        # Test NaN value (line 461)
        assert not dataset._feature_matches_selector(
            {'building': pd.NA}, {'building': '*'}
        )

    def test_feature_matches_selector_json_properties(self) -> None:
        """Test feature matching with JSON string properties (from GeoDataFrame)."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Test with JSON string properties (as might come from GeoDataFrame)
        json_props = {'properties': '{"building": "yes", "floors": "3"}'}
        assert dataset._feature_matches_selector(json_props, {'building': '*'})

    def test_get_label_with_precomputed(self) -> None:
        """Test get_label method with pre-computed labels."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Test with pre-computed label
        feature_with_label = {'properties': {'building': 'yes', 'label': 5}}
        assert dataset.get_label(feature_with_label) == 5

        # Test fallback to computation
        feature_without_label = {'properties': {'building': 'yes'}}
        assert dataset.get_label(feature_without_label) == 1

    def test_parse_overpass_response_with_labels(
        self, mock_download_and_integrity: None, multi_channel_params: dict[str, Any]
    ) -> None:
        """Test that _parse_overpass_response adds pre-computed labels."""
        dataset = OpenStreetMap(
            bbox=multi_channel_params['bbox'],
            paths=multi_channel_params['root'],
            classes=multi_channel_params['classes'],
            download=False,
        )

        # Mock OSM response
        mock_response = {
            'elements': [
                {
                    'type': 'way',
                    'id': 123,
                    'geometry': [
                        {'lat': 48.8566, 'lon': 2.3522},
                        {'lat': 48.8568, 'lon': 2.3524},
                        {'lat': 48.8568, 'lon': 2.3522},
                        {'lat': 48.8566, 'lon': 2.3522},
                    ],
                    'tags': {'building': 'residential'},
                },
                {
                    'type': 'way',
                    'id': 456,
                    'geometry': [
                        {'lat': 48.8566, 'lon': 2.3521},
                        {'lat': 48.8567, 'lon': 2.3523},
                        {'lat': 48.8567, 'lon': 2.3521},
                        {'lat': 48.8566, 'lon': 2.3521},
                    ],
                    'tags': {'amenity': 'restaurant'},
                },
            ]
        }

        gdf = dataset._parse_overpass_response(mock_response)

        assert len(gdf) == 2
        assert 'label' in gdf.columns
        assert gdf.iloc[0]['label'] == 1  # Building -> channel 1
        assert gdf.iloc[1]['label'] == 2  # Amenity -> channel 2

    def test_check_empty_classes(
        self,
        mock_download_and_integrity: None,
        multi_channel_params: dict[str, Any],
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test warning for empty channels."""
        # Create mock GDF with only building features, no amenity or highway
        mock_gdf = gpd.GeoDataFrame(
            {'building': ['yes'], 'label': [1], 'geometry': [Point(2.3523, 48.8567)]}
        )

        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        dataset = OpenStreetMap(
            bbox=multi_channel_params['bbox'],
            paths=multi_channel_params['root'],
            classes=multi_channel_params['classes'],
            download=False,
        )

        # Warning should appear on first query (lazy initialization)
        with pytest.warns(UserWarning, match="Class 'amenity' .* has no geometries"):
            dataset[dataset.bounds]

        # Second query should not trigger warning
        with warnings.catch_warnings():
            warnings.simplefilter('error')  # Turn warnings into errors
            dataset[dataset.bounds]  # Should not raise (no warning)

    def test_len(self, dataset: OpenStreetMap) -> None:
        """Test __len__ method."""
        # The fixture dataset should have test data
        assert len(dataset) == 1

    def test_plot_multi_feature_types(
        self,
        mock_download_and_integrity: None,
        multi_channel_params: dict[str, Any],
        monkeypatch: MonkeyPatch,
    ) -> None:
        """Test plotting with multiple feature types."""
        # Mock GDF with multiple feature types including background (label=0)
        mock_gdf = gpd.GeoDataFrame(
            {
                'building': ['yes', None, None, None],
                'highway': [None, 'primary', None, None],
                'amenity': [None, None, 'restaurant', 'cafe'],
                'geometry': [
                    Point(2.3521, 48.8566),
                    Point(2.3522, 48.8567),
                    Point(2.3523, 48.8568),
                    Point(2.3524, 48.8569),
                ],
                'label': [
                    1,
                    3,
                    2,
                    0,
                ],  # Updated labels: building=1, amenity=2, highway=3, background=0
            }
        )

        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        dataset = OpenStreetMap(
            bbox=multi_channel_params['bbox'],
            paths=multi_channel_params['root'],
            classes=multi_channel_params['classes'],
            download=False,
        )

        sample = {'mask': torch.zeros((10, 10))}  # Typical sample format
        fig = dataset.plot(sample)
        assert fig is not None

    def test_multichannel_dataset_functionality(
        self, multi_channel_params: dict[str, Any]
    ) -> None:
        """Test end-to-end functionality with multi-channel test data."""
        dataset = OpenStreetMap(
            bbox=multi_channel_params['bbox'],
            paths=multi_channel_params['root'],
            classes=multi_channel_params['classes'],
            download=False,
        )

        # Test that it loads successfully
        assert len(dataset) >= 0

        # Test querying
        sample = dataset[dataset.bounds]
        assert isinstance(sample, dict)

        if 'mask' in sample:
            mask = sample['mask']
            assert isinstance(mask, torch.Tensor)
            unique_labels = torch.unique(mask)
            # Should have background (0) and potentially other labels
            assert 0 in unique_labels

    def test_feature_matches_selector_json_decode_error(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        """Test feature matching with JSON decode error fallback."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Test with invalid JSON string properties (should fallback to original props)
        invalid_json_props = {'properties': 'invalid_json{'}
        # This should not raise an error but return False since the props don't match
        result = dataset._feature_matches_selector(
            invalid_json_props, {'building': '*'}
        )
        assert not result

    def test_plot_with_bounds_filtering(
        self, mock_download_and_integrity: None, monkeypatch: MonkeyPatch
    ) -> None:
        """Test plot method with bounds filtering."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Create mock GeoDataFrame with data
        mock_gdf = gpd.GeoDataFrame(
            {'building': ['yes'], 'label': [1], 'geometry': [Point(2.3523, 48.8567)]}
        )
        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Get proper sample from dataset with bounds
        bounds = (slice(2.3522, 2.3524), slice(48.8566, 48.8568), slice(None))
        sample = dataset[bounds]
        dataset.plot(sample)

    def test_plot_with_legend(
        self, mock_download_and_integrity: None, monkeypatch: MonkeyPatch
    ) -> None:
        """Test plot method showing legend."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Create mock GeoDataFrame with building data
        mock_gdf = gpd.GeoDataFrame(
            {'building': ['yes'], 'geometry': [Point(2.3523, 48.8567)], 'label': [1]}
        )
        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)
        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)
        sample = {'mask': torch.zeros((10, 10))}
        dataset.plot(sample)

    def test_plot_prediction(
        self, mock_download_and_integrity: None, monkeypatch: MonkeyPatch
    ) -> None:
        """Test plot method with prediction."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Create mock GeoDataFrame with data
        mock_gdf = gpd.GeoDataFrame(
            {'building': ['yes'], 'label': [1], 'geometry': [Point(2.3523, 48.8567)]}
        )
        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)

        # Get sample and add prediction
        query = dataset.bounds
        sample = dataset[query]
        sample['prediction'] = sample['mask'].clone()
        dataset.plot(sample, suptitle='Prediction')

    def test_plot_title_formatting(
        self, mock_download_and_integrity: None, monkeypatch: MonkeyPatch
    ) -> None:
        """Test plot method title formatting."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)

        # Create mock GeoDataFrame
        mock_gdf = gpd.GeoDataFrame({'geometry': []})
        monkeypatch.setattr('geopandas.read_file', lambda *_, **__: mock_gdf)

        channels = [{'name': 'building', 'selector': [{'building': '*'}]}]
        dataset = OpenStreetMap(bbox=bbox, paths=root, classes=channels, download=False)
        sample = {'mask': torch.zeros((10, 10))}
        dataset.plot(sample)
