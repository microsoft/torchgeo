# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.


import os

import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin


def generate_solar_plants_brazil_dataset(root: str) -> None:
    """Generate dummy data for train, val, and test splits."""

    transform = from_origin(-48.0, -15.0, 0.0001, 0.0001)
    crs = 'EPSG:4326'

    for split in ['train', 'val', 'test']:
        input_dir = os.path.join(root, split, 'input')
        label_dir = os.path.join(root, split, 'labels')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)

        # Create dummy image
        image_data = torch.randint(0, 2000, (4, 32, 32)).numpy().astype(np.float32)
        image_path = os.path.join(input_dir, 'img(1).tif')
        with rasterio.open(
            image_path,
            'w',
            driver='GTiff',
            height=32,
            width=32,
            count=4,
            dtype='float32',
            transform=transform,
            crs=crs,
        ) as dst:
            dst.write(image_data)

        # Create dummy mask
        mask_data = (torch.rand(32, 32) > 0.5).numpy().astype(np.uint8)
        mask_path = os.path.join(label_dir, 'target(1).tif')
        with rasterio.open(
            mask_path,
            'w',
            driver='GTiff',
            height=32,
            width=32,
            count=1,
            dtype='uint8',
            transform=transform,
            crs=crs,
        ) as dst:
            dst.write(mask_data, 1)
