#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import hashlib
import json
from typing import Any

# Building polygon with pre-computed label
building_geojson = {
    'type': 'FeatureCollection',
    'features': [
        {
            'type': 'Feature',
            'properties': {
                'building': 'residential',
                'osm_id': 12345,
                'osm_type': 'way',
                'label': 1,  # Pre-computed label
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

multi_channel_fixture_geojson = {
    'type': 'FeatureCollection',
    'features': [
        # Building feature - label 1 (first channel: building)
        {
            'type': 'Feature',
            'properties': {
                'building': 'yes',
                'osm_id': 12345,
                'osm_type': 'way',
                'label': 1,
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
        },
        # Amenity feature - label 2 (second channel: amenity)
        {
            'type': 'Feature',
            'properties': {
                'amenity': 'restaurant',
                'osm_id': 34567,
                'osm_type': 'node',
                'label': 2,
            },
            'geometry': {'type': 'Point', 'coordinates': [2.3523, 48.8567]},
        },
        # Highway feature - label 3 (third channel: highway)
        {
            'type': 'Feature',
            'properties': {
                'highway': 'primary',
                'osm_id': 23456,
                'osm_type': 'way',
                'label': 3,
            },
            'geometry': {
                'type': 'LineString',
                'coordinates': [[2.3520, 48.8565], [2.3525, 48.8570]],
            },
        },
    ],
}


def get_single_class_filename(
    class_name: str,
    bbox: tuple[float, float, float, float] = (2.3520, 48.8565, 2.3525, 48.8570),
) -> str:
    """Get filename for single class."""
    classes = [{'name': class_name, 'selector': [{class_name: '*'}]}]
    return get_classes_filename(classes, bbox)


def get_classes_filename(
    classes: list[dict[str, Any]],
    bbox: tuple[float, float, float, float] = (2.3520, 48.8565, 2.3525, 48.8570),
) -> str:
    cache_key = {'bbox': bbox, 'classes': classes}
    cache_str = json.dumps(cache_key, sort_keys=True)
    cache_hash = hashlib.md5(cache_str.encode()).hexdigest()[:16]
    return f'osm_features_{cache_hash}.geojson'


# --- STANDARDIZED FIXTURE DATA ---

# 1. common_test_params fixture - single building class
common_test_params_classes = [{'name': 'building', 'selector': [{'building': '*'}]}]
with open(get_classes_filename(common_test_params_classes), 'w') as f:
    json.dump(building_geojson, f)

# 2. multi_channel_params fixture - building, amenity, highway classes
multi_channel_params_classes = [
    {'name': 'building', 'selector': [{'building': '*'}]},
    {'name': 'amenity', 'selector': [{'amenity': '*'}]},
    {'name': 'highway', 'selector': [{'highway': '*'}]},
]


with open(get_classes_filename(multi_channel_params_classes), 'w') as f:
    json.dump(multi_channel_fixture_geojson, f)

# 4. Custom query test (mixed selectors in single class)
mixed_features_class = [
    {'name': 'mixed_features', 'selector': [{'building': '*'}, {'leisure': 'park'}]}
]
with open(get_classes_filename(mixed_features_class), 'w') as f:
    json.dump(multi_channel_fixture_geojson, f)  # Reuse same data

# 5. Multiple selectors test
mixed_selector_class = [
    {'name': 'mixed', 'selector': [{'building': '*'}, {'leisure': 'park'}]}
]
with open(get_classes_filename(mixed_selector_class), 'w') as f:
    json.dump(multi_channel_fixture_geojson, f)  # Reuse same data
