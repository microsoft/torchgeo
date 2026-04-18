#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import os
import shutil
import zipfile

import numpy as np
import rasterio
from PIL import Image as PILImage
from rasterio.crs import CRS
from rasterio.transform import Affine

np.random.seed(0)

SIZE = 64  # image width/height

wkt = """
PROJCS["Projection: Transverse Mercator; Datum: WGS84; Ellipsoid: WGS84",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0],
        UNIT["degree",0.0174532925199433,
            AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Transverse_Mercator"],
    PARAMETER["latitude_of_origin",0],
    PARAMETER["central_meridian",19],
    PARAMETER["scale_factor",0.9993],
    PARAMETER["false_easting",500000],
    PARAMETER["false_northing",-5300000],
    UNIT["metre",1,
        AUTHORITY["EPSG","9001"]],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]
"""

dtype = np.uint8
kwargs = {
    'driver': 'GTiff',
    'dtype': 'uint8',
    'crs': CRS.from_wkt(wkt),
    'transform': Affine(0.25, 0.0, 280307.7499987148, 0.0, -0.25, 394546.9999900842),
    'height': SIZE,
    'width': SIZE,
}
filename = 'M-33-20-D-c-4-2'

# Remove old data
zipfilename = 'landcover.ai.v1.zip'
for fn in ['train.txt', 'val.txt', 'test.txt', 'split.py', zipfilename]:
    if os.path.exists(fn):
        os.remove(fn)
for directory in ['images', 'masks', 'output']:
    if os.path.exists(directory):
        shutil.rmtree(directory)

# Create images
os.makedirs('images')
Z = np.random.randint(np.iinfo(dtype).max, size=(SIZE, SIZE), dtype=dtype)
with rasterio.open(
    os.path.join('images', f'{filename}.tif'), 'w', count=3, **kwargs
) as f:
    for i in range(1, 4):
        f.write(Z, i)

# Create masks
os.makedirs('masks')
Z = np.random.randint(4, size=(SIZE, SIZE), dtype=dtype)
with rasterio.open(
    os.path.join('masks', f'{filename}.tif'), 'w', count=1, **kwargs
) as f:
    f.write(Z, 1)

# Create train/val/test splits
files = ['M-33-20-D-c-4-2_0', 'M-33-20-D-c-4-2_1']
for split in ['train', 'val', 'test']:
    with open(f'{split}.txt', 'w') as f:
        for file in files:
            f.write(f'{file}\n')

# Create output directory with pre-chipped images using PIL instead of cv2
os.makedirs('output', exist_ok=True)

# Load the images and masks with rasterio and save as jpg/png
with rasterio.open(os.path.join('images', f'{filename}.tif')) as src:
    img_data = src.read()  # Read all bands
    # Convert from CxHxW to HxWxC for PIL
    img_data = np.transpose(img_data, (1, 2, 0))

with rasterio.open(os.path.join('masks', f'{filename}.tif')) as src:
    mask_data = src.read(1)  # Read single band

# Save pre-chipped versions
for i in range(2):
    # Save image as JPEG
    img = PILImage.fromarray(img_data.astype(np.uint8))
    img.save(os.path.join('output', f'{filename}_{i}.jpg'))

    # Save mask as PNG
    mask = PILImage.fromarray(mask_data.astype(np.uint8))
    mask.save(os.path.join('output', f'{filename}_{i}_m.png'))

# Compress data - include the pre-chipped output folder
with zipfile.ZipFile(zipfilename, 'w') as f:
    for file in [
        'images/M-33-20-D-c-4-2.tif',
        'masks/M-33-20-D-c-4-2.tif',
        'train.txt',
        'val.txt',
        'test.txt',
    ]:
        f.write(file, arcname=file)
    # Add output directory files
    for file in files:
        f.write(f'output/{file}.jpg', arcname=f'output/{file}.jpg')
        f.write(f'output/{file}_m.png', arcname=f'output/{file}_m.png')
