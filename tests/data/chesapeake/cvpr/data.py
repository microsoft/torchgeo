#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil

import numpy as np
import rasterio
from rasterio import Affine
from rasterio.crs import CRS

# Match the original tile's footprint exactly so sampling behavior (patch
# extraction, out-of-bounds handling, etc.) is unaffected by this fixture
# being synthetic rather than real imagery.
SIZE = 512
CRS_ = CRS.from_epsg(26918)
TRANSFORM = Affine(1.0, 0.0, 451549.0, 0.0, -1.0, 4316628.0)

TILE_DIR = 'de_1m_2013_extended-debuffered-test_tiles'
TILE = 'm_3807504_ne_18_1'

# NLCD land cover codes, a small representative subset of the full class list
NLCD_VALUES = [0, 11, 21, 41, 82, 95]

# Random pixels are generated on a small block and tiled to fill the full
# raster. This keeps the files small (a repeating pattern compresses well)
# while still exercising the full dtype/shape/value range of each layer.
BLOCK = 32

np.random.seed(0)


def write_raster(
    path: str, count: int, dtype: str, values: np.ndarray | None = None
) -> None:
    profile = {
        'driver': 'GTiff',
        'dtype': dtype,
        'count': count,
        'width': SIZE,
        'height': SIZE,
        'crs': CRS_,
        'transform': TRANSFORM,
        'compress': 'deflate',
    }
    shape = (count, BLOCK, BLOCK)
    if values is None:
        if dtype == 'float32':
            block = np.random.rand(*shape).astype(dtype)
        else:
            block = np.random.randint(np.iinfo(dtype).max + 1, size=shape).astype(dtype)
    else:
        block = np.random.choice(values, size=shape)
    data = np.tile(block, (1, SIZE // BLOCK, SIZE // BLOCK))

    with rasterio.open(path, 'w', **profile) as dst:
        dst.write(data)


def create_tile(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)

    # NAIP aerial imagery: 4-band (RGB + NIR) 8-bit
    for layer in ('naip-new', 'naip-old'):
        write_raster(os.path.join(directory, f'{TILE}_{layer}.tif'), 4, 'uint8')

    # Landsat 8 leaf-on/off composites: 9-band float32 surface reflectance
    for layer in ('landsat-leaf-on', 'landsat-leaf-off'):
        write_raster(os.path.join(directory, f'{TILE}_{layer}.tif'), 9, 'float32')

    # Chesapeake Bay high-resolution land cover: single-band, 7 classes
    write_raster(
        os.path.join(directory, f'{TILE}_lc.tif'),
        1,
        'uint8',
        np.arange(7, dtype='uint8'),
    )

    # NLCD land cover: single-band, subset of real class codes
    write_raster(
        os.path.join(directory, f'{TILE}_nlcd.tif'),
        1,
        'uint8',
        np.array(NLCD_VALUES, dtype='uint8'),
    )

    # Microsoft building footprints: single-band binary mask
    write_raster(
        os.path.join(directory, f'{TILE}_buildings.tif'),
        1,
        'uint8',
        np.arange(2, dtype='uint8'),
    )

    # Prior over land cover classes, distributed separately from the rest
    write_raster(
        os.path.join(
            directory, f'{TILE}_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif'
        ),
        4,
        'uint8',
    )


def create_archives(directory: str) -> None:
    prior_layer = f'{TILE}_prior_from_cooccurrences_101_31_no_osm_no_buildings.tif'
    prior_path = os.path.join(directory, prior_layer)

    # The base archive is distributed without the prior layer
    base_dir = os.path.join('base_archive', directory)
    os.makedirs(base_dir, exist_ok=True)
    for filename in os.listdir(directory):
        if filename != prior_layer:
            shutil.copy(os.path.join(directory, filename), base_dir)
    shutil.copy('spatial_index.geojson', 'base_archive')
    shutil.make_archive('cvpr_chesapeake_landcover', 'zip', 'base_archive')
    shutil.rmtree('base_archive')

    # The prior layer is distributed as a separate extension archive
    extension_dir = os.path.join('extension_archive', directory)
    os.makedirs(extension_dir, exist_ok=True)
    shutil.copy(prior_path, extension_dir)
    shutil.make_archive(
        'cvpr_chesapeake_landcover_prior_extension', 'zip', 'extension_archive'
    )
    shutil.rmtree('extension_archive')


if __name__ == '__main__':
    create_tile(TILE_DIR)
    create_archives(TILE_DIR)
