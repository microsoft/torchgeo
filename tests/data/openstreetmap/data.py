"""Test data for OpenStreetMap dataset tests."""

import hashlib
import json
from pathlib import Path
from typing import Any

import geopandas as gpd


def get_overpass_response() -> dict[str, Any]:
    """Get mock Overpass API response with test data.
    
    Returns:
        Dictionary containing mock Overpass API response with buildings and amenities.
    """
    return {
        "elements": [
            {
                "type": "way",
                "id": 123456,
                "tags": {"building": "yes", "name": "Test Building", "building:levels": "3"},
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
                "id": 456789,
                "tags": {"highway": "residential", "name": "Test Street"},
                "geometry": [
                    {"lat": 40.0002, "lon": -74.0002},
                    {"lat": 40.0008, "lon": -73.9998},
                    {"lat": 40.001, "lon": -73.999}
                ]
            }
        ]
    }


def create_test_geojson() -> None:
    """Create test GeoJSON file from the Overpass response data."""
    data_dir = Path(__file__).parent
    osm_dir = data_dir / "osm"
    osm_dir.mkdir(exist_ok=True)
    geojson_path = osm_dir / "test_data.geojson"

    # Building polygon
    building_coords = [[-74.0, 40.0], [-74.0, 40.001], [-73.999, 40.001], [-73.999, 40.0], [-74.0, 40.0]]
    building_feature = {
        "type": "Feature",
        "properties": {
            "building": "yes",
            "name": "Test Building", 
            "building:levels": "3",
            "osm_id": 123456,
            "osm_type": "way"
        },
        "geometry": {
            "type": "Polygon", 
            "coordinates": [building_coords]
        }
    }
    # Amenity point
    cafe_feature = {
        "type": "Feature",
        "properties": {
            "amenity": "cafe",
            "name": "Test Cafe",
            "osm_id": 789012,
            "osm_type": "node"
        },
        "geometry": {
            "type": "Point",
            "coordinates": [-73.9995, 40.0005]
        }
    }
    # Highway linestring
    highway_coords = [[-74.0002, 40.0002], [-73.9998, 40.0008], [-73.999, 40.001]]
    highway_feature = {
        "type": "Feature", 
        "properties": {
            "highway": "residential",
            "name": "Test Street",
            "osm_id": 456789,
            "osm_type": "way"
        },
        "geometry": {
            "type": "LineString",
            "coordinates": highway_coords
        }
    }
    features = [building_feature, cafe_feature, highway_feature]
    # Create GeoJSON structure
    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    # Write to file
    with open(geojson_path, 'w') as f:
        json.dump(geojson, f, indent=2)

    print(f"Created test GeoJSON file: {geojson_path}")


def get_test_gdf() -> gpd.GeoDataFrame:
    """Get test GeoDataFrame from the GeoJSON file.
    
    Returns:
        GeoDataFrame containing test OpenStreetMap data.
    """
    data_dir = Path(__file__).parent
    geojson_path = data_dir / "osm" / "test_data.geojson"

    if not geojson_path.exists():
        create_test_geojson()
    
    return gpd.read_file(geojson_path)


def create_test_data_files() -> None:
    """Create test data files for testing functionality."""
    data_dir = Path(__file__).parent
    osm_dir = data_dir / "osm"
    osm_dir.mkdir(exist_ok=True)
    
    # Create deterministic data filename based on known parameters
    # This mimics what OpenStreetMap._get_data_filename would generate
    # for bbox (-74.0, 40.0, -73.9, 40.1) and feature_type 'building'
    cache_key = {
        'bbox': (-74.0, 40.0, -73.9, 40.1),
        'feature_type': 'building',
        'custom_query': None,
    }
    cache_str = json.dumps(cache_key, sort_keys=True)
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:16]
    data_filename = f'osm_building_{cache_hash}.geojson'
    
    # Create the actual data file using our test GeoJSON
    data_path = osm_dir / data_filename
    
    # Copy our test data as the data file
    test_geojson_path = osm_dir / "test_data.geojson"
    if test_geojson_path.exists():
        import shutil
        shutil.copy2(test_geojson_path, data_path)
        print(f"Created test data file: {data_path}")

    # Create additional test files for different feature types and bboxes
    test_bboxes = [
        (-74.0, 40.0, -73.9, 40.1),  # NYC test area
        (2.349, 48.853, 2.351, 48.855),  # Paris test area used in tests
    ]
    
    for bbox in test_bboxes:
        for feature_type in ['building', 'highway', 'amenity', 'leisure', 'natural']:
            # Regular feature type files
            cache_key = {
                'bbox': bbox,
                'feature_type': feature_type,
                'custom_query': None,
            }
            cache_str = json.dumps(cache_key, sort_keys=True)
            cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:16]
            feature_filename = f'osm_{feature_type}_{cache_hash}.geojson'
            feature_path = osm_dir / feature_filename
            
            if test_geojson_path.exists():
                shutil.copy2(test_geojson_path, feature_path)
                print(f"Created test data file for {feature_type} at {bbox}: {feature_path}")
                
            # Also create files for custom queries used in tests
            # For amenity with cafe custom query (used in multiple tests)
            if feature_type == 'building':  # Create for building feature type instead since that's the default
                custom_query = "[out:json]; node[amenity=cafe]({{bbox}}); out;"
                cache_key_custom = {
                    'bbox': bbox,
                    'feature_type': feature_type,  # Keep as 'building' since that's the default
                    'custom_query': custom_query,
                }
                cache_str_custom = json.dumps(cache_key_custom, sort_keys=True)
                cache_hash_custom = hashlib.md5(cache_str_custom.encode()).hexdigest()[:16]
                custom_filename = f'osm_{feature_type}_{cache_hash_custom}.geojson'
                custom_path = osm_dir / custom_filename
                
                if test_geojson_path.exists():
                    shutil.copy2(test_geojson_path, custom_path)
                    print(f"Created test data file for custom query {feature_type} at {bbox}: {custom_path}")

    # Also create a sample JSON for mock responses
    json_sample_path = osm_dir / "sample_response.json"
    with open(json_sample_path, 'w') as f:
        json.dump(get_overpass_response(), f, indent=2)
    
    print(f"Created sample JSON response: {json_sample_path}")




def main() -> None:
    """Create all test data files."""
    print("Creating OpenStreetMap test data...")
    create_test_geojson()
    create_test_data_files()
    print("Test data creation complete!")


if __name__ == "__main__":
    main()