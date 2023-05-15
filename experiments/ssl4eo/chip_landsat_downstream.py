#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import os

import numpy as np
import rasterio
from rasterio.io import DatasetReader
from rasterio.vrt import WarpedVRT
from rasterio.windows import from_bounds


def retrieve_mask_chip(img_src: DatasetReader, mask_src: DatasetReader) -> np.ndarray:
    """Retrieve the mask for a given landsat image.

    Args:
        img_src: input image for which to find a corresponding chip
        mask_src: CRS aligned mask from which to retrieve a chip corresponding to
            img_src

    Returns:
        mask array
    """
    query = img_src.bounds
    out_width = round((query.right - query.left) / img_src.res[0])
    out_height = round((query.top - query.bottom) / img_src.res[1])
    out_shape = (1, out_height, out_width)
    mask = mask_src.read(
        out_shape=out_shape,
        window=from_bounds(
            query.left, query.bottom, query.right, query.top, img_src.transform
        ),
    )
    return mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Can be same directory for in-place compression
    parser.add_argument(
        "--landsat_dir", help="directory to recursively search for files"
    )
    parser.add_argument("--mask_path", help="path to downstream task mask to chip")
    parser.add_argument("--suffix", default=".tif", help="file suffix")
    # masks will be added as separate band to the *landsat_dir*

    args = parser.parse_args()

    mask_root_dir = os.path.join(args.landsat_dir, "masks")
    os.makedirs(mask_root_dir, exist_ok=True)

    # find all files in landsat dir
    paths = sorted(
        glob.glob(
            os.path.join(args.landsat_dir, "imgs", "*", "*", f"*{args.suffix}"),
            recursive=True,
        )
    )

    mask_src = rasterio.open(args.mask_path)
    img_crs = rasterio.open(paths[0]).crs

    if mask_src.crs != img_crs:
        mask_src = WarpedVRT(mask_src, crs=img_crs)

    for img_path in paths[0:100]:
        img_src = rasterio.open(img_path)

        # retrieve mask
        mask = retrieve_mask_chip(img_src, mask_src)

        # match directory structure
        mask_num_dir = os.path.join(
            mask_root_dir, os.path.dirname(img_path).split("/")[-2]
        )
        os.makedirs(mask_num_dir, exist_ok=True)
        mask_id_dir = os.path.join(
            mask_num_dir, os.path.dirname(img_path).split("/")[-1]
        )
        os.makedirs(mask_id_dir, exist_ok=True)

        # write mask tif
        profile = img_src.profile
        profile["count"] = 1

        with rasterio.open(
            os.path.join(mask_id_dir, "mask.tif"), "w", **profile
        ) as dst:
            dst.write(mask)
