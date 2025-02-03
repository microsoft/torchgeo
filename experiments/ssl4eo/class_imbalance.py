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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('roots', nargs='+', help='directories to search for files')
    parser.add_argument('--suffix', default='.tif', help='file suffix')
    parser.add_argument('--sort', action='store_true', help='sort by class frequency')
    parser.add_argument(
        '--weights', action='store_true', help='print weights instead of ratios'
    )
    parser.add_argument(
        '--total-classes', type=int, default=256, help='total number of classes'
    )
    parser.add_argument(
        '--keep-classes',
        type=float,
        default=1,
        help='keep classes with percentage higher than this',
    )
    parser.add_argument(
        '--ignore-index', type=int, default=0, help='fill value to ignore'
    )
    parser.add_argument('--num-workers', type=int, default=10, help='number of threads')
    args = parser.parse_args()

    def class_counts(path: str) -> np.typing.NDArray[np.float64]:
        """Calculate the number of values in each class.

        Args:
            path: Path to an image file.

        Returns:
            Counts of each class.
        """
        global args

        counts = np.zeros(args.total_classes)
        with rio.open(path, 'r') as src:
            x = src.read()
            unique, unique_counts = np.unique(x, return_counts=True)
            counts[unique] = unique_counts

        return counts

    paths = []
    for root in args.roots:
        paths.extend(
            glob.glob(os.path.join(root, '**', f'*{args.suffix}'), recursive=True)
        )

    if args.num_workers > 0:
        counts = thread_map(class_counts, paths, max_workers=args.num_workers)
    else:
        counts = []
        for path in tqdm(paths):
            counts.append(class_counts(path))

    counts = np.sum(counts, axis=0)

    if 0 <= args.ignore_index < args.total_classes:
        counts[args.ignore_index] = 0

    if args.sort:
        indices = np.argsort(counts)
        indices = indices[::-1]
        counts = counts[indices]
    else:
        indices = np.arange(args.total_classes)

    counts /= np.sum(counts)
    keep = counts > args.keep_classes / 100
    indices = indices[keep]
    counts = counts[keep]

    if args.weights:
        counts = 1 / counts
        counts /= np.sum(counts)

    print(indices)
    print(counts)
