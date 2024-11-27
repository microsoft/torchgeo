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
    # Can be same directory for in-place compression
    parser.add_argument('src_dir', help='directory to recursively search for files')
    parser.add_argument('dst_dir', help='directory to save compressed files in')
    parser.add_argument('--suffix', default='.tif', help='file suffix')
    # Could be min/max, 2%/98%, mean ± 2 * std, etc.
    parser.add_argument(
        '--min', nargs='+', type=float, required=True, help='minimum range'
    )
    parser.add_argument(
        '--max', nargs='+', type=float, required=True, help='maximum range'
    )
    parser.add_argument('--num-workers', type=int, default=10, help='number of threads')
    args = parser.parse_args()

    args.min = np.array(args.min)[:, np.newaxis, np.newaxis]
    args.max = np.array(args.max)[:, np.newaxis, np.newaxis]

    def compress(src_path: str) -> None:
        """Rescale, convert to uint8, and compress an image.

        Args:
            src_path: Path to an image file.
        """
        global args
        dst_path = src_path.replace(args.src_dir, args.dst_dir)
        dst_dir = os.path.dirname(dst_path)
        os.makedirs(dst_dir, exist_ok=True)
        with rio.open(src_path, 'r') as src:
            x = src.read()

            x = (x - args.min) / (args.max - args.min)

            # 0-1 -> 0-255
            x = np.clip(x * 255, 0, 255).astype(np.uint8)

            profile = src.profile
            profile['dtype'] = 'uint8'
            profile['compress'] = 'lzw'
            profile['predictor'] = 2
            with rio.open(dst_path, 'w', **profile) as dst:
                for i, band in enumerate(dst.indexes):
                    dst.write(x[i], band)

    paths = glob.glob(
        os.path.join(args.src_dir, '**', f'*{args.suffix}'), recursive=True
    )

    if args.num_workers > 0:
        thread_map(compress, paths, max_workers=args.num_workers)
    else:
        for path in tqdm(paths):
            compress(path)
