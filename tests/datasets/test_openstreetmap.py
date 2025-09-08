# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import pathlib
from unittest.mock import patch

import pytest

from torchgeo.datasets import DatasetNotFoundError, OpenStreetMap


class TestOpenStreetMap:
    @pytest.fixture
    def dataset(self) -> OpenStreetMap:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)  # Small Paris bbox
        return OpenStreetMap(bbox=bbox, paths=root, download=False)

    def test_init_no_download(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        # Use a different bbox that won't have test data
        bbox = (0.0, 0.0, 0.001, 0.001)
        with pytest.raises(DatasetNotFoundError):
            OpenStreetMap(bbox=bbox, paths=root, download=False)

    @patch('torchgeo.datasets.openstreetmap.OpenStreetMap._download_data')
    def test_init_with_download(self, mock_download) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap') 
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        mock_download.return_value = None
        
        with patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=True)
            mock_download.assert_called_once()
            assert dataset.bbox == bbox
            assert dataset.feature_type == 'building'

    def test_feature_types(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            # Test different feature types
            dataset_building = OpenStreetMap(bbox=bbox, paths=root, feature_type='building')
            assert dataset_building.feature_type == 'building'
            
            dataset_highway = OpenStreetMap(bbox=bbox, paths=root, feature_type='highway')  
            assert dataset_highway.feature_type == 'highway'
            
            dataset_amenity = OpenStreetMap(bbox=bbox, paths=root, feature_type='amenity')
            assert dataset_amenity.feature_type == 'amenity'

    def test_custom_query(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        custom_query = '[out:json]; way["building"]({{bbox}}); out geom;'
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, custom_query=custom_query)
            assert dataset.custom_query == custom_query

    def test_build_overpass_query(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, feature_type='building')
            query = dataset._build_overpass_query()
            assert 'building' in query
            assert '48.8565,2.352,48.857,2.3525' in query  # bbox format: south,west,north,east

    def test_get_data_filename(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, feature_type='building')
            filename = dataset._get_data_filename()
            assert filename.suffix == '.geojson'
            assert 'osm_building' in filename.name

    @patch('geopandas.read_file')
    def test_getitem(self, mock_read_file) -> None:
        import geopandas as gpd
        from shapely.geometry import Point
        
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        # Mock empty GeoDataFrame
        mock_gdf = gpd.GeoDataFrame({'geometry': [Point(2.3523, 48.8567)]})
        mock_read_file.return_value = mock_gdf
        
        with patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            dataset = OpenStreetMap(bbox=bbox, paths=root)
            # Use dataset bounds for querying
            sample = dataset[dataset.bounds]
            
            assert 'vector' in sample
            mock_read_file.assert_called()

    def test_plot(self) -> None:
        import geopandas as gpd
        
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True), \
             patch('geopandas.read_file') as mock_read_file:
            
            # Test with empty data
            mock_gdf = gpd.GeoDataFrame({'geometry': []})
            mock_read_file.return_value = mock_gdf
            
            dataset = OpenStreetMap(bbox=bbox, paths=root)
            sample = {'vector': mock_gdf}
            fig = dataset.plot(sample)
            
            assert fig is not None

    def test_paths_as_iterable(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            # Test paths as iterable (covers line 108)
            dataset = OpenStreetMap(bbox=bbox, paths=[root], feature_type='building')
            assert dataset.root.name == 'openstreetmap'

    def test_check_integrity(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'):
            # Test _check_integrity when file doesn't exist
            dataset = OpenStreetMap.__new__(OpenStreetMap)  # Create instance without __init__
            dataset.bbox = bbox
            dataset.feature_type = 'nonexistent'
            dataset.custom_query = None
            dataset.root = pathlib.Path(root)
            assert not dataset._check_integrity()

    def test_custom_query_bbox_replacement(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            # Test custom query with {{bbox}} replacement (covers lines 158-159)
            custom_query = '[out:json]; way["building"]({{bbox}}); out geom;'
            dataset = OpenStreetMap(bbox=bbox, paths=root, custom_query=custom_query)
            query = dataset._build_overpass_query()
            assert '48.8565,2.352,48.857,2.3525' in query
            assert '{{bbox}}' not in query

    def test_highway_query_generation(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            # Test highway query generation (covers lines 175-182)
            dataset = OpenStreetMap(bbox=bbox, paths=root, feature_type='highway')
            query = dataset._build_overpass_query()
            assert 'way["highway"]' in query
            assert 'relation["highway"]' not in query  # highway only uses ways

    def test_amenity_query_generation(self) -> None:
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            # Test amenity query generation (covers lines 183-192)
            dataset = OpenStreetMap(bbox=bbox, paths=root, feature_type='amenity')
            query = dataset._build_overpass_query()
            assert 'node["amenity"]' in query
            assert 'way["amenity"]' in query
            assert 'relation["amenity"]' in query

    def test_plot_with_suptitle(self) -> None:
        import geopandas as gpd
        
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True), \
             patch('geopandas.read_file') as mock_read_file:
            
            mock_gdf = gpd.GeoDataFrame({'geometry': []})
            mock_read_file.return_value = mock_gdf
            
            dataset = OpenStreetMap(bbox=bbox, paths=root)
            sample = {'vector': mock_gdf}
            fig = dataset.plot(sample, suptitle='Test Title')
            
            assert fig is not None

    def test_plot_without_titles(self) -> None:
        import geopandas as gpd
        
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True), \
             patch('geopandas.read_file') as mock_read_file:
            
            mock_gdf = gpd.GeoDataFrame({'geometry': []})
            mock_read_file.return_value = mock_gdf
            
            dataset = OpenStreetMap(bbox=bbox, paths=root)
            sample = {'vector': mock_gdf}
            fig = dataset.plot(sample, show_titles=False)
            
            assert fig is not None

    def test_rate_limiting(self) -> None:
        """Test rate limiting functionality."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=False)
            
            # Reset class variable for test
            OpenStreetMap._last_request_time = 0.0
            
            import time
            start_time = time.time()
            dataset._rate_limit()
            dataset._rate_limit()
            end_time = time.time()
            
            # Should take at least the minimum interval (with small tolerance for timing)
            assert end_time - start_time >= dataset._min_request_interval * 0.9

    def test_parse_overpass_response(self) -> None:
        """Test parsing of Overpass API response."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=False)
            
            # Mock OSM API response with different element types
            mock_osm_response = {
                'elements': [
                    {
                        'type': 'node',
                        'id': 123,
                        'lat': 48.8566,
                        'lon': 2.3523,
                        'tags': {'amenity': 'restaurant', 'name': 'Test Restaurant'}
                    },
                    {
                        'type': 'way',
                        'id': 456,
                        'geometry': [
                            {'lat': 48.8565, 'lon': 2.3520},
                            {'lat': 48.8570, 'lon': 2.3525}
                        ],
                        'tags': {'highway': 'primary'}
                    }
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

    def test_parse_overpass_response_empty(self) -> None:
        """Test parsing empty Overpass API response.""" 
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=False)
            
            # Empty response
            empty_response = {'elements': []}
            gdf = dataset._parse_overpass_response(empty_response)
            
            assert len(gdf) == 0
            assert gdf.crs == 'EPSG:4326'

    def test_element_to_geometry_node(self) -> None:
        """Test converting OSM node to Point geometry."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=False)
            
            # Test node element (covers lines 292-296)
            node_element = {
                'type': 'node',
                'lat': 48.8566,
                'lon': 2.3523
            }
            
            geom = dataset._element_to_geometry(node_element)
            assert geom is not None
            assert geom.geom_type == 'Point'
            assert geom.x == 2.3523
            assert geom.y == 48.8566

    def test_element_to_geometry_way_linestring(self) -> None:
        """Test converting OSM way to LineString geometry."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=False)
            
            # Test way element → LineString (covers lines 298-306)
            way_element = {
                'type': 'way',
                'geometry': [
                    {'lat': 48.8565, 'lon': 2.3520},
                    {'lat': 48.8570, 'lon': 2.3525}
                ]
            }
            
            geom = dataset._element_to_geometry(way_element)
            assert geom is not None
            assert geom.geom_type == 'LineString'

    def test_element_to_geometry_way_polygon(self) -> None:
        """Test converting OSM way to Polygon geometry."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=False)
            
            # Test way element → Polygon (closed way, covers lines 302-305)
            way_element = {
                'type': 'way', 
                'geometry': [
                    {'lat': 48.8566, 'lon': 2.3522},
                    {'lat': 48.8566, 'lon': 2.3524},
                    {'lat': 48.8568, 'lon': 2.3524},
                    {'lat': 48.8568, 'lon': 2.3522},
                    {'lat': 48.8566, 'lon': 2.3522}  # Closed polygon
                ]
            }
            
            geom = dataset._element_to_geometry(way_element)
            assert geom is not None
            assert geom.geom_type == 'Polygon'

    def test_element_to_geometry_invalid(self) -> None:
        """Test handling invalid elements."""
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=False)
            
            # Test invalid elements (covers line 308 - return None)
            invalid_elements = [
                {'type': 'node'},  # Missing lat/lon
                {'type': 'way'},   # Missing geometry
                {'type': 'unknown'}, # Unknown type
                {'type': 'way', 'geometry': [{'lat': 48.8565, 'lon': 2.3520}]}  # Single point way
            ]
            
            for element in invalid_elements:
                geom = dataset._element_to_geometry(element)
                assert geom is None

    def test_getitem_empty_data(self) -> None:
        """Test __getitem__ when no data is found in query area."""
        import geopandas as gpd
        
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True), \
             patch('geopandas.read_file') as mock_read_file:
            
            # Mock empty GeoDataFrame
            empty_gdf = gpd.GeoDataFrame({'geometry': []})
            mock_read_file.return_value = empty_gdf
            
            dataset = OpenStreetMap(bbox=bbox, paths=root, download=False)
            sample = dataset[dataset.bounds]
            
            # Should create empty GeoDataFrame (covers line 340)
            assert 'vector' in sample
            assert len(sample['vector']) == 0
            assert sample['vector'].crs == 'EPSG:4326'

    def test_plot_without_vector_key(self) -> None:
        """Test plot method when sample doesn't have 'vector' key."""
        import geopandas as gpd
        from shapely.geometry import Point
        
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True), \
             patch('geopandas.read_file') as mock_read_file:
            
            # Mock GeoDataFrame with actual data
            mock_gdf = gpd.GeoDataFrame({'geometry': [Point(2.3523, 48.8567)]})
            mock_read_file.return_value = mock_gdf
            
            dataset = OpenStreetMap(bbox=bbox, paths=root)
            
            # Sample without 'vector' key - should read original file (covers line 367)
            sample = {'some_other_key': 'value'}
            fig = dataset.plot(sample)
            
            assert fig is not None
            mock_read_file.assert_called()

    def test_plot_with_actual_data(self) -> None:
        """Test plot method with actual geometry data."""
        import geopandas as gpd
        from shapely.geometry import Point
        
        root = os.path.join('tests', 'data', 'openstreetmap')
        bbox = (2.3520, 48.8565, 2.3525, 48.8570)
        
        with patch.object(OpenStreetMap, '_download_data'), \
             patch.object(OpenStreetMap, '_check_integrity', return_value=True):
            
            dataset = OpenStreetMap(bbox=bbox, paths=root)
            
            # Sample with actual geometry data (covers line 373)
            gdf_with_data = gpd.GeoDataFrame({
                'geometry': [Point(2.3523, 48.8567)],
                'amenity': ['restaurant']
            })
            sample = {'vector': gdf_with_data}
            
            fig = dataset.plot(sample)
            assert fig is not None