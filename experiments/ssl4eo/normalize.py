#!/usr/bin/env python3

import argparse
import glob
import os

import numpy as np
import rasterio as rio


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to recursively search for files")
    parser.add_argument("--ext", default="tif", help="file extension")
    parser.add_argument("--nan", type=float, default=0, help="fill value")
    args = parser.parse_args()

    s0 = 0
    s1 = 0
    s2 = 0
    for path in glob.iglob(
        os.path.join(args.directory, "**", f"*.{args.ext}"), recursive=True
    ):
        with rio.open(path) as f:
            x = f.read()
            x[x == args.nan] = np.NAN
            s0 += np.count_nonzero(~np.isnan(x), axis=(1, 2))
            s1 += np.nansum(x, axis=(1, 2))
            s2 += np.nansum(x**2, axis=(1, 2))

    # https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
    m = s1 / s0
    s = np.sqrt((s0 * s2 - s1**2) / (s0 * (s0 - 1)))

    print("mean:")
    print(repr(m))

    print("std dev:")
    print(repr(s))
