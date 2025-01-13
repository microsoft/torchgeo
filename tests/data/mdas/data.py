#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin

# Set the random seed for reproducibility
np.random.seed(0)

# Define the root directory, dataset name, subareas, and modalities based on mdas.py
root_dir = '.'
ds_root_name = 'Augsburg_data_4_publication'
subareas = ['sub_area_1', 'sub_area_2', 'sub_area_3']
modalities = [
    '3K_DSM',
    '3K_RGB',
    'HySpex',
    'EeteS_EnMAP_10m',
    'EeteS_EnMAP_30m',
    'EeteS_Sentinel_2_10m',
    'Sentinel_1',
    'Sentinel_2',
    'osm_buildings',
    'osm_landuse',
    'osm_water',
]

landuse_class_codes = [
    -2147483647,  # no label
    7201,  # forest
    7202,  # park
    7203,  # residential
    7204,  # industrial
    7205,  # farm
    7206,  # cemetery
    7207,  # allotments
    7208,  # meadow
    7209,  # commercial
    7210,  # nature reserve
    7211,  # recreation ground
    7212,  # retail
    7213,  # military
    7214,  # quarry
    7215,  # orchard
    7217,  # scrub
    7218,  # grass
    7219,  # heath
]

# Remove existing dummy data if it exists
dataset_path = os.path.join(root_dir, ds_root_name)
if os.path.exists(dataset_path):
    shutil.rmtree(dataset_path)


def create_dummy_geotiff(
    path: str,
    num_bands: int = 3,
    width: int = 32,
    height: int = 32,
    dtype: np.dtype = np.uint16,
    binary: bool = False,
    landuse: bool = False,
) -> None:
    """Create a dummy GeoTIFF file."""
    crs = CRS.from_epsg(32632)
    transform = from_origin(0, 0, 1, 1)

    if binary:
        data = np.random.randint(0, 2, size=(num_bands, height, width)).astype(dtype)
    elif landuse:
        num_pixels = num_bands * height * width
        no_label_ratio = 0.1
        num_no_label = int(no_label_ratio * num_pixels)
        num_labels = num_pixels - num_no_label
        landuse_values = np.random.choice(landuse_class_codes[1:], size=num_labels)
        no_label_values = np.full(num_no_label, landuse_class_codes[0], dtype=dtype)
        combined = np.concatenate([landuse_values, no_label_values])
        np.random.shuffle(combined)
        data = combined.reshape((num_bands, height, width)).astype(dtype)
    else:
        # Generate random data for other modalities
        data = np.random.randint(0, 255, size=(num_bands, height, width)).astype(dtype)

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with rasterio.open(
        path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=num_bands,
        dtype=dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data)


# Create directory structure and dummy data
for subarea in subareas:
    # Format the subarea name for filenames, as in mdas.py _format_subarea method
    parts = subarea.split('_')
    subarea_formatted = parts[0] + '_' + parts[1] + parts[2]  # e.g., 'sub_area1'

    subarea_dir = os.path.join(root_dir, ds_root_name, subarea)

    for modality in modalities:
        filename = f'{modality}_{subarea_formatted}.tif'
        file_path = os.path.join(subarea_dir, filename)

        if modality in ['osm_buildings', 'osm_water']:
            create_dummy_geotiff(file_path, num_bands=1, dtype=np.uint8, binary=True)
        elif modality == 'osm_landuse':
            create_dummy_geotiff(file_path, num_bands=1, dtype=np.float64, landuse=True)
        elif modality == 'HySpex':
            create_dummy_geotiff(file_path, num_bands=368, dtype=np.int16)
        elif modality in ['EeteS_EnMAP_10m', 'EeteS_EnMAP_30m']:
            create_dummy_geotiff(file_path, num_bands=242, dtype=np.uint16)
        elif modality == 'Sentinel_1':
            create_dummy_geotiff(file_path, num_bands=2, dtype=np.float32)
        elif modality in ['Sentinel_2', 'EeteS_Sentinel_2_10m']:
            create_dummy_geotiff(file_path, num_bands=13, dtype=np.uint16)
        elif modality == '3K_DSM':
            create_dummy_geotiff(file_path, num_bands=1, dtype=np.float32)
        elif modality == '3K_RGB':
            create_dummy_geotiff(file_path, num_bands=3, dtype=np.uint8)

print(f'Dummy MDAS dataset created at {os.path.join(root_dir, ds_root_name)}')

# Create a zip archive of the dataset directory
zip_filename = f'{ds_root_name}.zip'
zip_path = os.path.join(root_dir, zip_filename)

shutil.make_archive(
    base_name=os.path.splitext(zip_path)[0],
    format='zip',
    root_dir='.',
    base_dir=ds_root_name,
)


def calculate_md5(filename: str) -> str:
    hash_md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


checksum = calculate_md5(zip_path)
print(f'MD5 checksum: {checksum}')
