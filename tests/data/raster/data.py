# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
from typing import Optional

import numpy as np
import rasterio as rio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject

RES = [2, 4, 8]
EPSG = [4087, 4326, 32631]
SIZE = 16


def write_raster(
    res: int = RES[0],
    epsg: int = EPSG[0],
    dtype: str = "uint8",
    path: Optional[str] = None,
) -> None:
    """Write a raster file.

    Args:
        res: Resolution.
        epsg: EPSG of file.
        dtype: Data type.
        path: File path.
    """
    size = SIZE // res
    profile = {
        "driver": "GTiff",
        "dtype": dtype,
        "count": 1,
        "crs": f"epsg:{epsg}",
        "transform": from_bounds(0, 0, SIZE, SIZE, size, size),
        "height": size,
        "width": size,
        "nodata": 0,
    }

    if path is None:
        name = f"res_{res}_epsg_{epsg}"
        path = os.path.join(name, f"{name}.tif")

    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)

    with rio.open(path, "w", **profile) as f:
        x = np.ones((1, size, size))
        f.write(x)


def reproject_raster(res: int, src_epsg: int, dst_epsg: int) -> None:
    """Reproject a raster file.

    Args:
        res: Resolution.
        src_epsg: EPSG of source file.
        dst_epsg: EPSG of destination file.
    """
    src_name = f"res_{res}_epsg_{src_epsg}"
    src_path = os.path.join(src_name, f"{src_name}.tif")
    with rio.open(src_path) as src:
        dst_crs = f"epsg:{dst_epsg}"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds
        )
        profile = src.profile.copy()
        profile.update(
            {"crs": dst_crs, "transform": transform, "width": width, "height": height}
        )
        dst_name = f"res_{res}_epsg_{dst_epsg}"
        os.makedirs(dst_name, exist_ok=True)
        dst_path = os.path.join(dst_name, f"{dst_name}.tif")
        with rio.open(dst_path, "w", **profile) as dst:
            reproject(
                source=rio.band(src, 1),
                destination=rio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst.transform,
                dst_crs=dst.crs,
            )


if __name__ == "__main__":
    for res in RES:
        src_epsg = EPSG[0]
        write_raster(res, src_epsg)

        for dst_epsg in EPSG[1:]:
            reproject_raster(res, src_epsg, dst_epsg)

    for dtype in ["uint16", "uint32"]:
        path = os.path.join(dtype, f"{dtype}.tif")
        write_raster(dtype=dtype, path=path)
        with open(os.path.join(dtype, "corrupted.tif"), "w") as f:
            f.write("not a tif file\n")
