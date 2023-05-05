#!/usr/bin/env python3

import argparse
import glob
import os
from multiprocessing.dummy import Pool

import numpy as np
import numpy.ma as ma
import rasterio as rio

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to recursively search for files")
    parser.add_argument("--suffix", default=".tif", help="file suffix")
    parser.add_argument("--nan", type=float, default=0, help="fill value")
    parser.add_argument("--num-workers", type=int, default=10, help="number of threads")
    parser.add_argument(
        "--chunksize", type=int, default=1000, help="size of process pool"
    )
    args = parser.parse_args()

    def calculate(path: str) -> tuple[int, float, float]:
        """Compute the count, sum, and sum of squares of an image.

        Args:
            path: Path to an image file.

        Returns:
            Count, sum, and sum of squares.
        """
        with rio.open(path) as f:
            x = f.read()
            y = ma.masked_equal(x, args.nan)
            s0 = np.count_nonzero(y, axis=(1, 2))
            s1 = np.sum(y, axis=(1, 2))
            s2 = np.sum(y**2, axis=(1, 2))
        return s0, s1, s2

    paths = glob.iglob(
        os.path.join(args.directory, "**", f"*{args.suffix}"), recursive=True
    )

    if args.num_workers > 0:
        with Pool(args.num_workers) as p:
            out = p.imap_unordered(calculate, paths, args.chunksize)
            s0s, s1s, s2s = zip(*out)
        s0: int = sum(s0s)
        s1: float = sum(s1s)
        s2: float = sum(s2s)
    else:
        s0 = s1 = s2 = 0
        for path in paths:
            s = calculate(path)
            s0 += s[0]
            s1 += s[1]
            s2 += s[2]

    # https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
    m = s1 / s0
    s = np.sqrt((s0 * s2 - s1**2) / (s0 * (s0 - 1)))

    print("mean:")
    print(repr(m))

    print("std dev:")
    print(repr(s))
