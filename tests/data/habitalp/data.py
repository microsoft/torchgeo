#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 64
NUM_CLASSES = 24
NUM_CHANGE_CLASSES = 9

np.random.seed(0)

profile = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'nodata': None,
    'crs': CRS.from_epsg(32633),
    'transform': Affine(1.0, 0.0, 500000.0, 0.0, -1.0, 5300000.0),
    'blockysize': 1,
    'tiled': False,
    'interleave': 'pixel',
    'height': SIZE,
    'width': SIZE,
}


def generate_rgb(path: str) -> None:
    """Generate fake RGB GeoTIFF."""
    p = profile.copy()
    p['count'] = 3
    data = np.random.randint(0, 255, size=(3, SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(path, mode='w', **p) as src:
        src.write(data)


def generate_cir(path: str) -> None:
    """Generate fake CIR (NIR, R, G) GeoTIFF."""
    p = profile.copy()
    p['count'] = 3
    data = np.random.randint(0, 255, size=(3, SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(path, mode='w', **p) as src:
        src.write(data)


def generate_terrain(path: str) -> None:
    """Generate fake terrain layer GeoTIFF."""
    p = profile.copy()
    p['count'] = 1
    p['dtype'] = 'float32'
    data = np.random.rand(1, SIZE, SIZE).astype(np.float32) * 100
    with rasterio.open(path, mode='w', **p) as src:
        src.write(data)


def generate_mask(path: str) -> None:
    """Generate fake segmentation mask GeoTIFF."""
    p = profile.copy()
    p['count'] = 1
    data = np.random.randint(0, NUM_CLASSES, size=(1, SIZE, SIZE), dtype=np.uint8)
    with rasterio.open(path, mode='w', **p) as src:
        src.write(data)


def generate_change_mask(path: str) -> None:
    """Generate fake multiclass change detection mask GeoTIFF."""
    p = profile.copy()
    p['count'] = 1
    data = np.random.randint(
        0, NUM_CHANGE_CLASSES, size=(1, SIZE, SIZE), dtype=np.uint8
    )
    with rasterio.open(path, mode='w', **p) as src:
        src.write(data)


if __name__ == '__main__':
    terrain_layers = [
        'dtm',
        'dsm',
        'ndsm',
        'slope',
        'aspect',
        'curvature',
        'planform_curvature',
        'profile_curvature',
        'roughness_terrain',
        'roughness_canopy',
        'tpi',
        'tri',
    ]

    os.makedirs('data_2003', exist_ok=True)
    os.makedirs('data_2013', exist_ok=True)
    os.makedirs('data_2020', exist_ok=True)
    os.makedirs('labels', exist_ok=True)

    generate_rgb('data_2003/aerial_rgb_2003_2007.tif')

    generate_rgb('data_2013/aerial_rgb_2013_2015.tif')
    generate_cir('data_2013/aerial_cir_2013_2015.tif')
    for layer in terrain_layers:
        generate_terrain(f'data_2013/{layer}.tif')

    generate_rgb('data_2020/aerial_rgb_2019_2021.tif')
    generate_cir('data_2020/aerial_cir_2019_2021.tif')
    for layer in terrain_layers:
        generate_terrain(f'data_2020/{layer}.tif')

    generate_mask('labels/classes_2003.tif')
    generate_mask('labels/classes_2013.tif')
    generate_mask('labels/classes_2020.tif')

    generate_change_mask('labels/habitalp_change_2003_2013.tif')
    generate_change_mask('labels/habitalp_change_2013_2020.tif')
