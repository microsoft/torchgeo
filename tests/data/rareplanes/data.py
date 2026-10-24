#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import json
import os

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_origin


def generate_test_data(root: str) -> None:
    """Generate fake RarePlanes data."""
    for split in ('train', 'test'):
        real = os.path.join(root, 'real', split)
        image_directory = os.path.join(real, 'PS-RGB_tiled')
        annotation_directory = os.path.join(real, 'geojson_aircraft_tiled')
        os.makedirs(image_directory, exist_ok=True)
        os.makedirs(annotation_directory, exist_ok=True)

        image_path = os.path.join(image_directory, 'scene.png')
        with rasterio.open(
            image_path,
            'w',
            driver='PNG',
            width=8,
            height=8,
            count=3,
            dtype=np.uint8,
            crs='EPSG:4326',
            transform=from_origin(-108, 46, 0.01, 0.01),
        ) as destination:
            destination.write(np.zeros((3, 8, 8), dtype=np.uint8))

        annotation = {
            'type': 'FeatureCollection',
            'features': [
                {
                    'type': 'Feature',
                    'properties': {'is_plane': 1},
                    'geometry': {
                        'type': 'Polygon',
                        'coordinates': [
                            [
                                [-107.99, 45.98],
                                [-107.96, 45.98],
                                [-107.96, 45.94],
                                [-107.99, 45.94],
                                [-107.99, 45.98],
                            ]
                        ],
                    },
                }
            ],
        }
        with open(os.path.join(annotation_directory, 'scene.geojson'), 'w') as file:
            json.dump(annotation, file)

        synthetic = os.path.join(root, 'synthetic', split)
        image_directory = os.path.join(synthetic, 'images')
        annotation_directory = os.path.join(synthetic, 'xmls')
        os.makedirs(image_directory, exist_ok=True)
        os.makedirs(annotation_directory, exist_ok=True)
        Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(
            os.path.join(image_directory, 'scene.png')
        )
        annotation = """<image>
<object><category0>Airplane</category0><bndbox2D>
<xmin>1</xmin><ymin>2</ymin><xmax>4</xmax><ymax>6</ymax>
</bndbox2D></object>
<object><category0>Airport</category0><bndbox2D>
<xmin>0</xmin><ymin>0</ymin><xmax>1</xmax><ymax>1</ymax>
</bndbox2D></object>
</image>"""
        with open(os.path.join(annotation_directory, 'scene.xml'), 'w') as file:
            file.write(annotation)


if __name__ == '__main__':
    generate_test_data(os.path.dirname(__file__))
