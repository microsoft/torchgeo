#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json

# Create minimal test data for OpenStreetMap dataset
# Small bounding box in Paris with basic feature types

# Building polygon
building_geojson = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'properties': {
                'building': 'residential',
                'osm_id': 12345,
                'osm_type': 'way',
            },
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [2.3522, 48.8566],
                        [2.3524, 48.8566],
                        [2.3524, 48.8568],
                        [2.3522, 48.8568],
                        [2.3522, 48.8566],
                    ]
                ],
            },
        }
    ],
}

# Highway linestring
highway_geojson = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'properties': {'highway': 'primary', 'osm_id': 23456, 'osm_type': 'way'},
            'geometry': {
                'type': 'LineString',
                'coordinates': [[2.3520, 48.8565], [2.3525, 48.8570]],
            },
        }
    ],
}

# Amenity point
amenity_geojson = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'properties': {
                'amenity': 'restaurant',
                'osm_id': 34567,
                'osm_type': 'node',
            },
            'geometry': {'type': 'Point', 'coordinates': [2.3523, 48.8567]},
        }
    ],
}

# Leisure point for testing generic feature types
leisure_geojson = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'properties': {'leisure': 'park', 'osm_id': 45678, 'osm_type': 'way'},
            'geometry': {
                'type': 'Polygon',
                'coordinates': [
                    [
                        [2.3521, 48.8566],
                        [2.3523, 48.8566],
                        [2.3523, 48.8567],
                        [2.3521, 48.8567],
                        [2.3521, 48.8566],
                    ]
                ],
            },
        }
    ],
}

# Empty geojson for testing no-data scenario
empty_geojson = {'type': 'FeatureCollection', 'features': []}

# Calculate expected filenames based on OpenStreetMap's naming scheme


def get_osm_filename(
    feature_type: str,
    bbox: tuple[float, float, float, float] = (2.3520, 48.8565, 2.3525, 48.8570),
    custom_query: str | None = None,
) -> str:
    cache_key = {
        'bbox': bbox,
        'feature_type': feature_type,
        'custom_query': custom_query,
    }
    cache_str = json.dumps(cache_key, sort_keys=True)
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:16]
    return f'osm_{feature_type}_{cache_hash}.geojson'


# Create the test files with correct OpenStreetMap naming
with open(get_osm_filename('building'), 'w') as f:
    json.dump(building_geojson, f)

with open(get_osm_filename('highway'), 'w') as f:
    json.dump(highway_geojson, f)

with open(get_osm_filename('amenity'), 'w') as f:
    json.dump(amenity_geojson, f)

with open(get_osm_filename('leisure'), 'w') as f:
    json.dump(leisure_geojson, f)

# Create file for custom query test
custom_query = '[out:json]; way["building"]({{bbox}}); out geom;'
with open(get_osm_filename('building', custom_query=custom_query), 'w') as f:
    json.dump(building_geojson, f)

# Also create old test files for backwards compatibility
with open('osm_building_test.geojson', 'w') as f:
    json.dump(building_geojson, f)

with open('osm_highway_test.geojson', 'w') as f:
    json.dump(highway_geojson, f)

with open('osm_amenity_test.geojson', 'w') as f:
    json.dump(amenity_geojson, f)

with open('osm_empty_test.geojson', 'w') as f:
    json.dump(empty_geojson, f)
