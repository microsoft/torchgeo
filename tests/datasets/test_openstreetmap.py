# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

"""Tests for OpenStreetMap dataset."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import matplotlib.pyplot as plt
import pytest
import shapely
from matplotlib.figure import Figure
from pyproj import CRS

from torchgeo.datasets import OpenStreetMap
from torchgeo.datasets.errors import DatasetNotFoundError

# Test constants for geometry tests
TEST_NODE_ELEMENT = {
    "type": "node",
    "id": 123456,
    "lat": 40.0,
    "lon": -74.0
}

TEST_WAY_POLYGON_ELEMENT = {
    "type": "way",
    "id": 123456,
    "geometry": [
        {"lat": 40.0, "lon": -74.0},
        {"lat": 40.001, "lon": -74.0},
        {"lat": 40.001, "lon": -73.999},
        {"lat": 40.0, "lon": -73.999},
        {"lat": 40.0, "lon": -74.0}  # closed polygon
    ]
}

TEST_WAY_LINESTRING_ELEMENT = {
    "type": "way", 
    "id": 123456,
    "geometry": [
        {"lat": 40.0, "lon": -74.0},
        {"lat": 40.001, "lon": -74.0},
        {"lat": 40.001, "lon": -73.999}  # open linestring
    ]
}

TEST_INVALID_ELEMENT = {
    "type": "invalid",
    "id": 123
}

TEST_INSUFFICIENT_WAY = {
    "type": "way",
    "id": 123,
    "geometry": [{"lat": 40.0, "lon": -74.0}]  # Only one coordinate
}


class TestOpenStreetMap:
    """Test OpenStreetMap dataset."""
    
    @pytest.fixture
    def dataset_root(self) -> Path:
        """Get path to test data directory."""
        return Path(__file__).parent.parent / "data" / "openstreetmap" / "osm"
    
    
    def teardown_method(self) -> None:
        """Clean up cache files after each test."""
        dataset_root = Path(__file__).parent.parent / "data" / "openstreetmap"
        osm_dir = dataset_root / "osm"
        
        # Only remove cache files created during tests, preserve permanent test data
        # Permanent test data: osm_cache_building_d2ca0726a4530331.geojson
        permanent_cache_files = {"osm_cache_building_d2ca0726a4530331.geojson"}
        
        if osm_dir.exists():
            for cache_file in osm_dir.glob("osm_cache_*.geojson"):
                if cache_file.name not in permanent_cache_files:
                    cache_file.unlink(missing_ok=True)
    
    @pytest.fixture 
    def mock_overpass_response(self) -> dict:
        """Create mock Overpass API response."""
        return {
            "elements": [
                {
                    "type": "way",
                    "id": 123456,
                    "tags": {"building": "yes", "name": "Test Building"},
                    "geometry": [
                        {"lat": 40.0, "lon": -74.0},
                        {"lat": 40.001, "lon": -74.0},
                        {"lat": 40.001, "lon": -73.999},
                        {"lat": 40.0, "lon": -73.999},
                        {"lat": 40.0, "lon": -74.0}
                    ]
                },
                {
                    "type": "node",
                    "id": 789012,
                    "lat": 40.0005,
                    "lon": -73.9995,
                    "tags": {"amenity": "cafe", "name": "Test Cafe"}
                },
                {
                    "type": "way",
                    "id": 345678,
                    "tags": {"highway": "residential", "name": "Test Road"},
                    "geometry": [
                        {"lat": 40.0, "lon": -74.0},
                        {"lat": 40.001, "lon": -73.999},
                        {"lat": 40.002, "lon": -73.998}
                    ]
                }
            ]
        }
    
    def test_init_default(self, dataset_root: Path) -> None:
        """Test dataset initialization with default parameters."""
        bbox = (2.349, 48.853, 2.351, 48.855)  # Default test bbox
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, download=False)
        assert dataset.root == dataset_root
        assert dataset.bbox == bbox
        assert dataset.feature_type == "building"
        assert dataset.custom_query is None
    
    def test_init_custom_params(self, dataset_root: Path) -> None:
        """Test dataset initialization with custom parameters."""
        custom_query = "[out:json]; node[amenity=cafe]({{bbox}}); out;"
        crs = CRS.from_epsg(3857)
        
        bbox = (2.349, 48.853, 2.351, 48.855)
        dataset = OpenStreetMap(
            bbox=bbox,
            paths=dataset_root,
            crs=crs,
            feature_type="amenity", 
            custom_query=custom_query,
            download=False  # Avoid network calls in tests
        )
        
        assert dataset.bbox == bbox
        assert dataset.crs == crs
        assert dataset.feature_type == "amenity"
        assert dataset.custom_query == custom_query
    
    def test_build_overpass_query_building(self, dataset_root: Path) -> None:
        """Test Overpass query building for buildings."""
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, feature_type="building", download=False)
        
        query = dataset._build_overpass_query()
        
        assert "[out:json]" in query
        assert 'way["building"]' in query
        assert 'relation["building"]' in query
        assert "40.0,-74.0,40.1,-73.9" in query  # overpass bbox format
    
    def test_build_overpass_query_highway(self, dataset_root: Path) -> None:
        """Test Overpass query building for highways."""
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, feature_type="highway", download=False)
        
        query = dataset._build_overpass_query()
        
        assert 'way["highway"]' in query
        assert 'node["highway"]' not in query  # highways are typically ways only
    
    def test_build_overpass_query_custom(self, dataset_root: Path) -> None:
        """Test Overpass query building with custom query."""
        custom_query = "[out:json]; node[amenity=cafe]({{bbox}}); out;"
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, custom_query=custom_query, download=False)
        
        query = dataset._build_overpass_query()
        
        assert query == "[out:json]; node[amenity=cafe](40.0,-74.0,40.1,-73.9); out;"
    
    def test_build_overpass_query_generic(self, dataset_root: Path) -> None:
        """Test Overpass query building for generic feature type."""
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, feature_type="natural", download=False)
        
        query = dataset._build_overpass_query()
        
        assert "[out:json]" in query
        assert 'node["natural"]' in query
        assert 'way["natural"]' in query  
        assert 'relation["natural"]' in query
        assert "40.0,-74.0,40.1,-73.9" in query
    
    def test_build_overpass_query_amenity(self, dataset_root: Path) -> None:
        """Test Overpass query building for amenity feature type."""
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, feature_type="amenity", download=False)
        
        query = dataset._build_overpass_query()
        
        assert "[out:json]" in query
        assert 'node["amenity"]' in query
        assert 'way["amenity"]' in query  
        assert 'relation["amenity"]' in query
        assert "40.0,-74.0,40.1,-73.9" in query
    
    def test_get_data_filename(self, dataset_root: Path) -> None:
        """Test data filename generation."""
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, feature_type="building", download=False)
        
        filename = dataset._get_data_filename()
        
        assert "osm_building_" in str(filename)
        assert str(filename).endswith(".geojson")
        
        # Same parameters should generate same filename
        dataset2 = OpenStreetMap(bbox=bbox, paths=dataset_root, feature_type="building", download=False)
        filename2 = dataset2._get_data_filename()
        assert filename == filename2
        
        # Different feature type should generate different filename
        dataset3 = OpenStreetMap(bbox=bbox, paths=dataset_root, feature_type="highway", download=False)
        filename3 = dataset3._get_data_filename()
        assert filename != filename3
    
    def test_element_to_geometry_node(self, dataset_root: Path) -> None:
        """Test converting OSM node to geometry."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        geom = dataset._element_to_geometry(TEST_NODE_ELEMENT)
        
        assert isinstance(geom, shapely.Point)
        assert geom.x == -74.0
        assert geom.y == 40.0
    
    def test_element_to_geometry_way_polygon(self, dataset_root: Path) -> None:
        """Test converting OSM closed way to polygon."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        geom = dataset._element_to_geometry(TEST_WAY_POLYGON_ELEMENT)
        
        assert isinstance(geom, shapely.Polygon)
        assert len(geom.exterior.coords) == 5  # including closure
    
    def test_element_to_geometry_way_linestring(self, dataset_root: Path) -> None:
        """Test converting OSM open way to linestring."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        geom = dataset._element_to_geometry(TEST_WAY_LINESTRING_ELEMENT)
        
        assert isinstance(geom, shapely.LineString)
        assert len(geom.coords) == 3
    
    def test_parse_overpass_response(self, dataset_root: Path, mock_overpass_response: dict) -> None:
        """Test parsing Overpass API response."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        gdf = dataset._parse_overpass_response(mock_overpass_response)
        
        assert len(gdf) == 3
        assert gdf.crs == CRS.from_epsg(4326)
        
        # Check first element (building polygon)
        assert gdf.iloc[0]['osm_type'] == 'way'
        assert gdf.iloc[0]['osm_id'] == 123456
        assert gdf.iloc[0]['building'] == 'yes'
        assert isinstance(gdf.iloc[0]['geometry'], shapely.Polygon)
        
        # Check second element (amenity point)
        assert gdf.iloc[1]['osm_type'] == 'node'
        assert gdf.iloc[1]['osm_id'] == 789012
        assert gdf.iloc[1]['amenity'] == 'cafe'
        assert isinstance(gdf.iloc[1]['geometry'], shapely.Point)
        
        # Check third element (highway linestring)
        assert gdf.iloc[2]['osm_type'] == 'way'
        assert gdf.iloc[2]['osm_id'] == 345678
        assert gdf.iloc[2]['highway'] == 'residential'
        assert isinstance(gdf.iloc[2]['geometry'], shapely.LineString)
    
    @patch('torchgeo.datasets.openstreetmap.urlopen')
    def test_download_data_success(self, mock_urlopen: Mock, dataset_root: Path, mock_overpass_response: dict) -> None:
        """Test successful data download."""
        # Create proper mock context manager
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_overpass_response).encode('utf-8')
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_response)
        mock_context.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_context
        
        # Use different bbox to avoid cached data
        bbox = (-75.0, 41.0, -74.9, 41.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, download=True)
        
        # Patch the rate limiting to avoid delays
        with patch.object(dataset, '_rate_limit'):
            dataset._download_data()
        
        assert mock_urlopen.called
        
        # Check data file was created  
        data_file = dataset._get_data_filename()
        assert data_file.exists()
        
        # Check the contents by reading the file
        import geopandas as gpd
        gdf = gpd.read_file(data_file)
        assert len(gdf) == 3  # building, cafe, highway from mock data
        
        # Clean up
        data_file.unlink(missing_ok=True)
    
    def test_download_data_from_cache(self, dataset_root: Path, mock_overpass_response: dict) -> None:
        """Test loading data from cache."""
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, download=False)
        
        # Create data file for testing
        data_filename = dataset._get_data_filename()
        data_path = dataset.root / data_filename
        
        # Create mock GeoDataFrame and save to data file
        gdf_to_cache = dataset._parse_overpass_response(mock_overpass_response)
        gdf_to_cache.to_file(data_path, driver='GeoJSON')
        
        # Check that data was loaded successfully by creating a new dataset instance
        # that can read the cached data
        dataset2 = OpenStreetMap(bbox=bbox, paths=dataset_root, download=False)
        assert dataset2._check_integrity()  # Should find the data file
    
    def test_download_data_no_download(self, dataset_root: Path) -> None:
        """Test behavior when download=False and no cache."""
        # Use different bbox that has no cached data
        bbox = (-80.0, 45.0, -79.9, 45.1)
        
        with pytest.raises(DatasetNotFoundError):
            OpenStreetMap(bbox=bbox, paths=dataset_root, download=False)
    
    @patch('torchgeo.datasets.openstreetmap.urlopen')
    def test_download_data_failure(self, mock_urlopen: Mock, dataset_root: Path) -> None:
        """Test download failure when all endpoints fail."""
        # Make all requests fail
        mock_urlopen.side_effect = Exception("Connection failed")
        
        # Use a different bbox that doesn't have cached data
        bbox = (-80.0, 45.0, -79.9, 45.1)
        
        with patch('torchgeo.datasets.openstreetmap.OpenStreetMap._rate_limit'):
            with pytest.raises(RuntimeError, match="All Overpass API endpoints failed"):
                OpenStreetMap(bbox=bbox, paths=dataset_root, download=True)
    
    def test_getitem_basic(self, dataset_root: Path) -> None:
        """Test basic __getitem__ functionality."""
        # Use test data that already exists
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, download=False)
        
        # Use spatial slicing - VectorDataset expects 2D spatial slices
        query = (slice(-74.0, -73.9), slice(40.0, 40.1))
        
        sample = dataset[query]
        
        assert 'vector' in sample
        assert 'crs' in sample
        assert 'bounds' in sample
        assert len(sample['vector']) == 3  # building, cafe, highway from test data
        assert sample['crs'] == CRS.from_epsg(4326)
    
    def test_crs_property(self, dataset_root: Path) -> None:
        """Test CRS property getter and setter."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        # Test default CRS
        assert dataset.crs == CRS.from_epsg(4326)
        
        # Test setting CRS
        new_crs = CRS.from_epsg(32618)
        dataset.crs = new_crs
        assert dataset.crs == new_crs
    
    def test_element_to_geometry_invalid(self, dataset_root: Path) -> None:
        """Test element to geometry conversion with invalid data."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        # Test invalid element
        geom = dataset._element_to_geometry(TEST_INVALID_ELEMENT)
        assert geom is None
        
        # Test way with insufficient coordinates
        geom = dataset._element_to_geometry(TEST_INSUFFICIENT_WAY)
        assert geom is None
    
    def test_disambiguate_slice_single(self, dataset_root: Path) -> None:
        """Test slice disambiguation with single slice."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        # Test single slice gets converted to tuple
        single_slice = slice(-74.0, -73.9)
        result = dataset._disambiguate_slice(single_slice)
        
        assert len(result) == 3
        assert result[0].start == -74.0
        assert result[0].stop == -73.9
    
    def test_disambiguate_slice_with_step(self, dataset_root: Path) -> None:
        """Test slice disambiguation with step parameter.""" 
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        # Test slice with step
        
        query_with_step = (slice(-74.0, -73.9, 0.01), slice(40.0, 40.1, 0.01))
        result = dataset._disambiguate_slice(query_with_step)
        
        assert result[0].step == 0.01
        assert result[1].step == 0.01
    
    def test_download_data_corrupted_cache(self, dataset_root: Path) -> None:
        """Test handling of corrupted cache file."""
        # Use different bbox to avoid conflicting with existing cache
        bbox = (-75.0, 41.0, -74.9, 41.1)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, download=False)
        
        # Create corrupted data file
        data_file = dataset._get_data_filename()
        with open(data_file, 'w') as f:
            f.write("invalid json content")
        
        # Should raise DatasetNotFoundError when data is corrupted and download=False
        with pytest.raises(DatasetNotFoundError):
            # Create a new instance which will try to load the corrupted file
            OpenStreetMap(bbox=bbox, paths=dataset_root, download=False)
    
    
    
    def test_getitem_crs_transformation(self, dataset_root: Path) -> None:
        """Test CRS transformation with UTM data."""
        
        # Create dataset with UTM CRS using regular cached data
        utm_crs = CRS.from_epsg(32618)  
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False, crs=utm_crs)
        
        # Create a mock GeoDataFrame with test data in EPSG:4326
        # This bypasses cache lookup issues and directly tests the CRS transformation logic
        from unittest.mock import patch
        
        import geopandas as gpd
        import shapely
        
        # Create test data in EPSG:4326 (like what comes from Overpass API)
        test_data = gpd.GeoDataFrame({
            'osm_id': [12345],
            'osm_type': ['way'],
            'building': ['yes'],
            'geometry': [shapely.Point(-74.0, 40.0)]
        }, crs='EPSG:4326')
        
        # Mock _get_data to return our test data
        with patch.object(dataset, '_get_data', return_value=test_data):
            # Query with UTM coordinates - the exact values don't matter for this test
            # as we're testing the transformation, not the cache lookup
            utm_query = (slice(585000, 586000), slice(4428000, 4429000))
            sample = dataset[utm_query]
        
        # Should return data in the target CRS
        assert sample['crs'] == utm_crs
        assert 'vector' in sample
        
        # Verify we got the test data and it's in the correct CRS
        assert len(sample['vector']) > 0
        assert sample['vector'].crs == utm_crs
        
        # Verify the data was transformed from EPSG:4326 to UTM 
        # Original point was at -74.0, 40.0 (NYC area)
        # In UTM Zone 18N this should be ~580,000-590,000 E, ~4,420,000-4,430,000 N
        point_geom = sample['vector'].iloc[0].geometry
        assert 580000 < point_geom.x < 590000  # x coordinate in UTM
        assert 4420000 < point_geom.y < 4430000  # y coordinate in UTM

    def test_getitem_with_transforms(self, dataset_root: Path) -> None:
        """Test __getitem__ with transforms applied."""
        def dummy_transform(sample: dict) -> dict:
            sample['transformed'] = True
            return sample
            
        bbox = (2.349, 48.853, 2.351, 48.855)
        dataset = OpenStreetMap(bbox=bbox, paths=dataset_root, download=False, transforms=dummy_transform)
        
        query = (slice(2.349, 2.351), slice(48.853, 48.855))
        result = dataset[query]
        
        assert 'transformed' in result
        assert result['transformed'] is True
    
    def test_rate_limiting(self, dataset_root: Path) -> None:
        """Test rate limiting functionality."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        # Reset class variable for test
        OpenStreetMap._last_request_time = 0.0
        
        import time
        start_time = time.time()
        dataset._rate_limit()
        dataset._rate_limit()
        end_time = time.time()
        
        # Should take at least the minimum interval (with small tolerance for timing)
        assert end_time - start_time >= dataset._min_request_interval * 0.9
    
    @patch('torchgeo.datasets.openstreetmap.urlopen')
    def test_plot_with_data(self, mock_urlopen: Mock, dataset_root: Path, mock_overpass_response: dict) -> None:
        """Test plot method with data."""
        # Create proper mock context manager
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(mock_overpass_response).encode('utf-8')
        mock_context = Mock()
        mock_context.__enter__ = Mock(return_value=mock_response)
        mock_context.__exit__ = Mock(return_value=None)
        mock_urlopen.return_value = mock_context
        
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=True)
        query = (slice(2.349, 2.351), slice(48.853, 48.855))
        # Patch the rate limiting to avoid delays
        with patch.object(dataset, '_rate_limit'):
            sample = dataset[query]
        
        # Test basic plot
        fig = dataset.plot(sample)
        assert isinstance(fig, Figure)
        plt.close()
        
        # Test plot with custom title
        fig = dataset.plot(sample, suptitle='Test Plot')
        assert isinstance(fig, Figure)
        plt.close()
        
        # Test plot without titles
        fig = dataset.plot(sample, show_titles=False)
        assert isinstance(fig, Figure)
        plt.close()
    
    def test_plot_empty_data(self, dataset_root: Path) -> None:
        """Test plot method with empty data."""
        dataset = OpenStreetMap(bbox=(2.349, 48.853, 2.351, 48.855), paths=dataset_root, download=False)
        
        # Create empty sample
        sample = {
            'vector': dataset._parse_overpass_response({'elements': []}),
            'crs': CRS.from_epsg(4326)
        }
        
        fig = dataset.plot(sample)
        assert isinstance(fig, Figure)
        plt.close()
    
    def test_plot_custom_query(self, dataset_root: Path) -> None:
        """Test plot method with custom query dataset."""
        custom_query = "[out:json]; node[amenity=cafe]({{bbox}}); out;"
        bbox = (-74.0, 40.0, -73.9, 40.1)
        dataset = OpenStreetMap(
            bbox=bbox,
            paths=dataset_root, 
            custom_query=custom_query,
            download=False
        )
        
        # Create empty sample for custom query
        sample = {
            'vector': dataset._parse_overpass_response({'elements': []}),
            'crs': CRS.from_epsg(4326)
        }
        
        fig = dataset.plot(sample, show_titles=True)
        assert isinstance(fig, Figure)
        plt.close()