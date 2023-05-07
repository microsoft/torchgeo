#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import os

import numpy as np
import rasterio as rio
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to recursively search for files")
    parser.add_argument("--suffix", default=".tif", help="file suffix")
    parser.add_argument("--num-workers", type=int, default=10, help="number of threads")
    args = parser.parse_args()

    def compute(path: str) -> "np.typing.NDArray[np.float32]":
        """Compute the minimum and maximum values in a dataset.

        Args:
            path: Path to an image file.

        Returns:
            Min, max, mean, and std dev of the image.
        """
        with rio.open(path) as f:
            out = np.zeros((f.count, 4), dtype=np.float32)
            for band in f.indexes:
                stats = f.statistics(band)
                out[band - 1] = (stats.min, stats.max, stats.mean, stats.std)
        return out

    paths = glob.glob(
        os.path.join(args.directory, "**", f"*{args.suffix}"), recursive=True
    )

    if args.num_workers > 0:
        out = np.array(thread_map(compute, paths, max_workers=args.num_workers))
    else:
        out_list = []
        for path in tqdm(paths):
            out_list.append(compute(path))
        out = np.array(out_list)

    minimum = np.amin(out[:, :, 0])
    maximum = np.amax(out[:, :, 1])

    mu_d = out[:, :, 2]
    mu = np.mean(mu_d, axis=0)
    sigma_d = out[:, :, 3]
    N_d = 264**2
    N = len(mu_d) * N_d

    # https://stats.stackexchange.com/a/442050/188076
    sigma = np.sqrt(
        np.sum(sigma_d**2 * (N_d - 1) + N_d * (mu - mu_d) ** 2, axis=0) / (N - 1),
        dtype=np.float32,
    )

    print("min:", minimum)
    print("max:", maximum)
    print("mean:", mu)
    print("std:", sigma)
