#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#os.environ['PROJ_LIB'] = r'E:\Programs\anaconda3\envs\gis\Library\share\proj'
import hashlib
import os
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import Affine

SIZE = 32
wkt = """
PROJCS["Albers Conical Equal Area",
    GEOGCS["WGS 84",
        DATUM["WGS_1984",
            SPHEROID["WGS 84",6378137,298.257223563,
                AUTHORITY["EPSG","7030"]],
            AUTHORITY["EPSG","6326"]],
        PRIMEM["Greenwich",0,AUTHORITY["EPSG","8901"]],
        UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
        AUTHORITY["EPSG","4326"]],
    PROJECTION["Albers_Conic_Equal_Area"],
    PARAMETER["latitude_of_center",23],
    PARAMETER["longitude_of_center",-96],
    PARAMETER["standard_parallel_1",29.5],
    PARAMETER["standard_parallel_2",45.5],
    PARAMETER["false_easting",0],
    PARAMETER["false_northing",0],
    UNIT["meters",1],
    AXIS["Easting",EAST],
    AXIS["Northing",NORTH]]
"""


np.random.seed(0)
files = ["South_America_Soybean_2002.tif", "South_America_Soybean_2021.tif"]

def create_file(path: str, dtype: str):
    """Create the testing file."""
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        #"crs": CRS.from_wkt(wkt),
        "transform": Affine(
            0.0002499999999999943131,
            0.0,
            -82.0005000000000024,
            0.0,
            -0.0002499999999999943131,
            0.0005000000000000,
        ),
        "height": SIZE,
        "width": SIZE,
        "compress": "lzw",
        "predictor": 2,
    }

    allowed_values = [0, 1]

    Z = np.random.choice(allowed_values, size=(SIZE, SIZE))

    with rasterio.open(path, "w", **profile) as src:
        src.write(Z, 1)

if __name__ == "__main__":
    dir = os.path.join(os.getcwd(), "SouthAmericaSoybean")
    print(dir)
    if os.path.exists(dir) and os.path.isdir(dir):
        shutil.rmtree(dir)

    os.makedirs(dir, exist_ok=True)

    for file in files:
        create_file(os.path.join(dir, file), dtype="int8")

    # Compress data
    shutil.make_archive("SouthAmericaSoybean", "zip", ".", dir)

    # Compute checksums
    with open("SouthAmericaSoybean.zip", "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"SouthAmericaSoybean.zip: {md5}")
