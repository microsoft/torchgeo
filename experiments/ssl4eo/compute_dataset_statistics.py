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

    def compute(path: str) -> tuple["np.typing.NDArray[np.float32]", int]:
        """Compute the min, max, mean, and std dev of a single image.

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
        return out, f.width * f.height

    paths = glob.glob(
        os.path.join(args.directory, "**", f"*{args.suffix}"), recursive=True
    )

    if args.num_workers > 0:
        out_tuple, size_tuple = list(
            zip(*thread_map(compute, paths, max_workers=args.num_workers))
        )
        out = np.array(out_tuple)
        sizes = np.array(size_tuple)
    else:
        out_list = []
        size_list = []
        for path in tqdm(paths):
            out, size = compute(path)
            out_list.append(out)
            size_list.append(size)
        out = np.array(out_list)
        sizes = np.array(size_list)

    assert len(np.unique(sizes)) == 1

    minimum = np.amin(out[:, :, 0], axis=0)
    maximum = np.amax(out[:, :, 1], axis=0)

    mu_d = out[:, :, 2]
    mu = np.mean(mu_d, axis=0)
    sigma_d = out[:, :, 3]
    N_d = sizes[0]
    N = len(mu_d) * N_d

    # https://stats.stackexchange.com/a/442050/188076
    sigma = np.sqrt(
        np.sum(sigma_d**2 * (N_d - 1) + N_d * (mu - mu_d) ** 2, axis=0) / (N - 1),
        dtype=np.float32,
    )

    np.set_printoptions(linewidth=2**8)
    print("min:", repr(minimum))
    print("max:", repr(maximum))
    print("mean:", repr(mu))
    print("std:", repr(sigma))
