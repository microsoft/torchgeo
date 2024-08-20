#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import glob
import os

import numpy as np
import rasterio as rio
from sklearn.decomposition import IncrementalPCA

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', help='directory to recursively search for files')
    parser.add_argument('--ext', default='tif', help='file extension')
    parser.add_argument('--scale', default=255, type=float, help='scale factor')
    args = parser.parse_args()

    transformer = IncrementalPCA(n_components=1)
    for path in glob.iglob(
        os.path.join(args.directory, '**', f'*.{args.ext}'), recursive=True
    ):
        with rio.open(path) as f:
            x = f.read().astype(np.float32)
            x /= args.scale
            x = np.transpose(x, (1, 2, 0))
            x = x.reshape((-1, x.shape[-1]))
            transformer.partial_fit(x)

    print('pca:', transformer.components_)
