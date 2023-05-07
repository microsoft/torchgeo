#!/usr/bin/env python3

import argparse
import glob
import os
import sys
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
    parser.add_argument(
        "--min-max",
        action="store_true",
        help="whether to calculate min/max or mean/std",
    )
    args = parser.parse_args()

    minimum = sys.maxsize
    maximum = -sys.maxsize
    s0 = 0
    s1 = 0.0
    s2 = 0.0

    def min_max(path: str) -> None:
        """Compute the minimum and maximum values in a dataset.

        Args:
            path: Path to an image file.
        """
        with rio.open(path) as f:
            for band in f.indexes:
                stats = f.statistics(band)
                global minimum, maximum
                minimum = stats.min if stats.min < minimum else minimum
                maximum = stats.max if stats.max > maximum else maximum

    def mean_std(path: str) -> None:
        """Compute the count, sum, and sum of squares of an image.

        Args:
            path: Path to an image file.
        """
        with rio.open(path) as f:
            x = f.read()
            y = ma.masked_equal(x, args.nan)  # type: ignore[no-untyped-call]
            global s0, s1, s2
            s0 += np.count_nonzero(y, axis=(1, 2))
            s1 += np.sum(y, axis=(1, 2))
            s2 += np.sum(y**2, axis=(1, 2))

    paths = glob.iglob(
        os.path.join(args.directory, "**", f"*{args.suffix}"), recursive=True
    )

    if args.num_workers > 0:
        with Pool(args.num_workers) as p:
            if args.min_max:
                p.map(min_max, paths, args.chunksize)
            else:
                p.map(mean_std, paths, args.chunksize)
    else:
        for path in paths:
            if args.min_max:
                min_max(path)
            else:
                mean_std(path)

    if args.min_max:
        print("min:", minimum)
        print("max:", maximum)
    else:
        # https://en.wikipedia.org/wiki/Standard_deviation#Rapid_calculation_methods
        m = s1 / s0
        s = np.sqrt((s0 * s2 - s1**2) / (s0 * (s0 - 1)))

        print("mean:", m)
        print("std dev:", s)
