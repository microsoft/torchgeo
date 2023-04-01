#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import hashlib
import os
import random
import shutil

import numpy as np
import rasterio
from rasterio.crs import CRS

SIZE = 64

np.random.seed(0)
random.seed(0)

wkt = """
PROJCRS["WGS 84 / UTM zone 19S",
    BASEGEOGCRS["WGS 84",
        DATUM["World Geodetic System 1984",
            ELLIPSOID["WGS 84",6378137,298.257223563,
                LENGTHUNIT["metre",1]]],
        PRIMEM["Greenwich",0,
            ANGLEUNIT["degree",0.0174532925199433]],
        ID["EPSG",4326]],
    CONVERSION["UTM zone 19S",
        METHOD["Transverse Mercator", ID["EPSG",9807]],
        PARAMETER["Latitude of natural origin",0,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8801]],
        PARAMETER["Longitude of natural origin",-69,
            ANGLEUNIT["degree",0.0174532925199433],
            ID["EPSG",8802]],
        PARAMETER["Scale factor at natural origin",0.9996,SCALEUNIT["unity",1],
            ID["EPSG",8805]],
        PARAMETER["False easting",500000,LENGTHUNIT["metre",1],
            ID["EPSG",8806]],
        PARAMETER["False northing",10000000,LENGTHUNIT["metre",1],
            ID["EPSG",8807]]],
    CS[Cartesian,2],
        AXIS["(E)",east,ORDER[1],LENGTHUNIT["metre",1]],
        AXIS["(N)",north,ORDER[2],LENGTHUNIT["metre",1]]]
"""

def create_file(path: str, dtype: str, num_channels: int) -> None:
    profile = {}
    profile["driver"] = "GTiff"
    profile["dtype"] = dtype
    profile["count"] = num_channels
    profile["crs"] = CRS.from_wkt(wkt),
    profile["transform"] = rasterio.transform.from_bounds(0, 0, 1, 1, 1, 1) # may need to change
    profile["height"] = SIZE
    profile["width"] = SIZE
    profile["compress"] = "lzw"
    profile["predictor"] = 2 # predictor??
    cmap = {
        0: (0, 0, 0),
        64: (64, 64, 64),
        128: (128, 128, 128),
        192: (192, 192, 192),
        255: (255, 255, 255),
    }

    Z = np.random.randint(size=(SIZE, SIZE), low=0, high=8)

    with rasterio.open(path, "w", **profile) as src:
        for i in range(1, profile["count"] + 1):
            src.write(Z, i)

        src.write_colormap(1, cmap)

dirs = [
    "austral", "boreal", "mid_latitude_north", "mid_latitude_south", 
    "polar_north", "polar_south", "subtropical_north", "subtropical_south", 
    "tropical"]
patches = [
    "p226_r98", "p108_r18", "p107_r34", "p100_r82",
    "p110_r12", "p10_r120", "p118_r40", "p101_r72", 
    "p111_r55"]
filenames = [
    "L71226098_09820011112_", "L71108018_01820010729_", "L71107034_03420010722_B10", "L71100082_08220011212_",
    "L71110012_01220010727_", "L71010120_12020011119_", "L71118040_04020010703_B10", "L71101072_07220011203_",
    "L71111055_05520010413_"]
bands = ["B10", "B20", "B30", "B40", "B50", "B61", "B62", "B70", "B80"]


if __name__ == "__main__":
    # Create images
    for i in range(len(dirs)):
        filename = dirs[i]+".tar.gz"

        # Remove old data
        if os.path.isdir(dir):
            shutil.rmtree(dir)
        
        os.makedirs(os.path.join(os.getcwd(), dirs[i]))

        for band in bands:
            create_file(
                os.path.join(dirs[i], patches[i], filenames[i], band, ".TIF"),
                dtype="uint8",
                num_channels=1,
            )
        
        # Compress data
        shutil.make_archive(dirs[i], "gztar", dirs[i])

        # Compute checksums
        with open(filename, "rb") as f:
            md5 = hashlib.md5(f.read()).hexdigest()
            print(f"{filename}: {md5}")

        shutil.rmtree(dir)
