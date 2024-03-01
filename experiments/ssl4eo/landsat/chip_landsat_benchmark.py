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
from tqdm import tqdm


def retrieve_mask_chip(
    img_src: DatasetReader, mask_src: DatasetReader
) -> "np.typing.NDArray[np.uint8]":
    """Retrieve the mask for a given landsat image.

    Args:
        img_src: input image for which to find a corresponding chip
        mask_src: CRS aligned mask from which to retrieve a chip
            corresponding to img_src

    Returns:
        mask array
    """
    out_shape = (1, *img_src.shape)
    mask_chip: "np.typing.NDArray[np.uint8]" = mask_src.read(
        out_shape=out_shape, window=from_bounds(*img_src.bounds, mask_src.transform)
    )

    # Copy nodata pixels from image to mask (Landsat 7 ETM+ SLC-off only)
    if "LE07" in img_src.files[0]:
        img_chip = img_src.read(1)
        mask_chip[0][img_chip == 0] = 0

    return mask_chip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--landsat-dir", help="directory to recursively search for files", required=True
    )
    parser.add_argument(
        "--mask-path", help="path to downstream task mask to chip", required=True
    )
    parser.add_argument(
        "--save-dir", help="directory where to save masks", required=True
    )
    parser.add_argument("--suffix", default=".tif", help="file suffix")
    args = parser.parse_args()

    paths = glob.glob(
        os.path.join(args.landsat_dir, "**", f"all_bands{args.suffix}"), recursive=True
    )

    if "nlcd" in args.mask_path:
        layer_name = "nlcd"
    else:
        layer_name = "cdl"

    for img_path in tqdm(paths):
        with (
            rasterio.open(img_path) as img_src,
            rasterio.open(args.mask_path) as mask_src,
        ):
            if mask_src.crs != img_src.crs:
                mask_src = WarpedVRT(mask_src, crs=img_src.crs)

            # retrieve mask
            mask = retrieve_mask_chip(img_src, mask_src)

            # directory structure mask <7-digit id>/<scene_id>/<cdl>_<year>.tif
            digit_id, scene_id = img_path.split(os.sep)[-3:-1]
            year = scene_id.split("_")[-1][:4]
            mask_dir = os.path.join(args.save_dir, digit_id, scene_id)
            os.makedirs(mask_dir, exist_ok=True)

            # write mask tif
            profile = img_src.profile
            profile["count"] = 1
            profile["dtype"] = mask_src.profile["dtype"]

            with rasterio.open(
                os.path.join(mask_dir, f"{layer_name}_{year}.tif"), "w", **profile
            ) as dst:
                dst.write(mask)
                dst.write_colormap(1, mask_src.colormap(1))
