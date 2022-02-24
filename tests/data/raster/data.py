# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os

import numpy as np
import rasterio
import rasterio.transform
from torchvision.datasets.utils import calculate_md5


def generate_test_data(fn: str) -> str:
    """Creates test data with uint32 datatype.

    Args:
        fn (str): Filename to write

    Returns:
        str: md5 hash of created archive
    """
    profile = {
        "driver": "GTiff",
        "dtype": "uint32",
        "count": 1,
        "crs": "epsg:4326",
        "transform": rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1),
        "height": 4,
        "width": 4,
        "compress": "lzw",
        "predictor": 2,
    }

    with rasterio.open(fn, "w", **profile) as f:
        f.write(np.random.randint(0, 256, size=(1, 4, 4)))

    md5: str = calculate_md5(fn)
    return md5


if __name__ == "__main__":
    md5_hash = generate_test_data(os.path.join(os.getcwd(), "test0.tif"))
    print(md5_hash)
