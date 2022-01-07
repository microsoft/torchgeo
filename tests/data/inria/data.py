import os

import numpy as np
import rasterio as rio
from rasterio.crs import CRS
from rasterio.transform import Affine


def write_data(path, img, driver, crs, transform):
    with rio.open(
        path,
        "w",
        driver=driver,
        height=img.shape[0],
        width=img.shape[1],
        count=3,
        dtype=img.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(1, dst.count + 1):
            dst.write(img, i)


def generate_test_data(root: str, n_samples=2) -> None:
    dtype = np.dtype("uint8")
    size = (64, 64)

    driver = "GTiff"
    transform = Affine(0.3, 0.0, 616500.0, 0.0, -0.3, 3345000.0)
    crs = CRS.from_epsg(26914)

    img_dir = os.path.join(root, "AerialImageDataset", "train", "images")
    lbl_dir = os.path.join(root, "AerialImageDataset", "train", "gt")
    timg_dir = os.path.join(root, "AerialImageDataset", "test", "images")

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    if not os.path.exists(lbl_dir):
        os.makedirs(lbl_dir)
    if not os.path.exists(timg_dir):
        os.makedirs(timg_dir)

    for i in range(n_samples):

        img = np.random.randint(np.iinfo(dtype).max, size=size, dtype=dtype)
        lbl = np.random.randint(np.iinfo(dtype).max, size=size, dtype=dtype)
        timg = np.random.randint(np.iinfo(dtype).max, size=size, dtype=dtype)

        img_path = os.path.join(img_dir, f"austin{i+1}.tif")
        lbl_path = os.path.join(lbl_dir, f"austin{i+1}.tif")
        timg_path = os.path.join(timg_dir, f"austin{i+10}.tif")

        write_data(img_path, img, driver, crs, transform)
        write_data(lbl_path, lbl, driver, crs, transform)
        write_data(timg_path, timg, driver, crs, transform)


if __name__ == "__main__":
    generate_test_data(".")
