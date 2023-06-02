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
    parser.add_argument("roots", nargs="+", help="directories to search for files")
    parser.add_argument("--suffix", default=".tif", help="file suffix")
    parser.add_argument("--sort", action="store_true", help="sort by class frequency")
    parser.add_argument(
        "--num-classes", type=int, default=256, help="number of classes"
    )
    parser.add_argument("--num-workers", type=int, default=10, help="number of threads")
    args = parser.parse_args()

    def class_ratios(path: str) -> "np.typing.NDArray[np.float32]":
        """Calculate the ratios of each class.

        Args:
            path: Path to an image file.

        Returns:
            Class-wise ratios.
        """
        global args

        out = np.zeros(args.num_classes, dtype=np.float32)
        with rio.open(path, "r") as src:
            x = src.read()
            unique, unique_counts = np.unique(x, return_counts=True)
            out[unique] = unique_counts / x.size

        return out

    paths = []
    for root in args.roots:
        paths.extend(
            glob.glob(os.path.join(root, "**", f"*{args.suffix}"), recursive=True)
        )

    if args.num_workers > 0:
        ratios = thread_map(class_ratios, paths, max_workers=args.num_workers)
    else:
        ratios = []
        for path in tqdm(paths):
            ratios.append(class_ratios(path))

    ratio = np.mean(ratios, axis=0)

    if args.sort:
        indices = np.argsort(ratio)
        indices = indices[::-1]
        ratio = ratio[indices]
    else:
        indices = np.arange(args.num_classes)

    keep = ratio > 0
    indices = indices[keep]
    ratio = ratio[keep]

    print(indices)
    print(ratio)
