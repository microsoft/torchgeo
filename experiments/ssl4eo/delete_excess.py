#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import os
import shutil

from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", help="directory to search for scenes")
    parser.add_argument("--num-workers", type=int, default=10, help="number of threads")
    parser.add_argument(
        "--length", type=int, default=250000, help="number of scenes to keep"
    )
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(args.root, "*")))
    paths = paths[args.length :]

    if args.num_workers > 0:
        thread_map(shutil.rmtree, paths, max_workers=args.num_workers)
    else:
        for path in tqdm(paths):
            shutil.rmtree(path)
