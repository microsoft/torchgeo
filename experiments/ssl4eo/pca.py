#!/usr/bin/env python3

import argparse
import glob
import os

# import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
from sklearn.decomposition import IncrementalPCA


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", help="directory to recursively search for files")
    parser.add_argument("--ext", default="tif", help="file extension")
    parser.add_argument("--nan", default=0, type=float, help="fill value")
    # parser.add_argument(
    #     "--mean", nargs="*", type=float, help="mean for normalization"
    # )
    # parser.add_argument(
    #     "--std", nargs="*", type=float, help="std dev for normalization"
    # )
    args = parser.parse_args()

    transformer = IncrementalPCA(n_components=1)
    for path in glob.iglob(
        os.path.join(args.directory, "**", f"*.{args.ext}"), recursive=True
    ):
        with rio.open(path) as f:
            x = f.read()
            x = np.transpose(x, (1, 2, 0))
            x = x.reshape((-1, x.shape[-1]))
            transformer.partial_fit(x)

    print("pca:", transformer.components_)

    # for path in glob.iglob(
    #     os.path.join(args.directory, "**", f"*.{args.ext}"), recursive=True
    # ):
    #     with rio.open(path) as f:
    #         x = f.read()
    #         gray = x * np.expand_dims(transformer.components_.flatten(), axis=(1, 2))
    #         gray = np.sum(gray, axis=0)
    #         plt.imshow(gray, cmap="gray")
    #         plt.show()
