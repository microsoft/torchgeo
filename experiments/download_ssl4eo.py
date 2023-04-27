#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Sample and download Satellite images with Google Earth Engine

#### run the script:

### Install and authenticate Google Earth Engine

### match and download pre-sampled locations
python download_ssl4eo.py \
    --save-path ./data \
    --collection COPERNICUS/S2 \
    --meta-cloud-name CLOUDY_PIXEL_PERCENTAGE \
    --cloud-pct 20 \
    --dates 2021-12-21 2021-09-22 2021-06-21 2021-03-20 \
    --radius 1320 \
    --bands B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12 \
    --crops 44 264 264 264 132 132 132 264 132 44 44 132 132 \
    --dtype uint16 \
    --num-workers 8 \
    --log-freq 100 \
    --match-file ./data/sampled_locations.csv \
    --indices-range 0 250000

### resume from interruption
python download_ssl4eo.py \
    -- ... \
    --resume ./data/checked_locations.csv \
    --indices-range 0 250000

## Example1: download Landsat-8, match pre-sampled locations
python download_ssl4eo.py \
    --save-path ./data \
    --collection LANDSAT/LC08/C02/T1_TOA \
    --meta-cloud-name CLOUD_COVER \
    --cloud-pct 20 \
    --dates 2021-12-21 2021-09-22 2021-06-21 2021-03-20 \
    --radius 1980 \
    --bands B1 B2 B3 B4 B5 B6 B7 B8 B9 B10 B11 \
    --crops 132 132 132 132 132 132 132 264 264 132 132 \
    --dtype float32 \
    --num-workers 8 \
    --log-freq 100 \
    --match-file ./data/sampled_locations.csv \
    --indices-range 0 250000

"""

import argparse
import csv
import json
import os
import time
import warnings
from collections import OrderedDict
from datetime import date, datetime, timedelta
from multiprocessing.dummy import Lock, Pool
from typing import Any, Optional

import ee
import numpy as np
import rasterio
import urllib3
from rasterio.transform import Affine

warnings.simplefilter("ignore", UserWarning)


def date2str(date: datetime) -> str:
    return date.strftime("%Y-%m-%d")


def get_period(date: datetime, days: int = 5) -> tuple[str, str, str, str]:
    date1 = date - timedelta(days=days / 2)
    date2 = date + timedelta(days=days / 2)
    date3 = date1 - timedelta(days=365)
    date4 = date2 - timedelta(days=365)
    return (
        date2str(date1),
        date2str(date2),
        date2str(date3),
        date2str(date4),
    )  # two-years buffer


"""get collection and remove clouds from ee"""


def mask_clouds(args: Any, image: ee.Image) -> ee.Image:
    qa = image.select(args.qa_band)
    cloudBitMask = 1 << args.qa_cloud_bit
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    return image.updateMask(mask)


def get_collection(
    collection_name: str, meta_cloud_name: str, cloud_pct: float
) -> ee.ImageCollection:
    collection = ee.ImageCollection(collection_name)
    collection = collection.filter(
        ee.Filter.And(
            ee.Filter.gte(meta_cloud_name, 0), ee.Filter.lte(meta_cloud_name, cloud_pct)
        )
    )
    # Uncomment the following line if you want to apply cloud masking.
    # collection = collection.map(mask_clouds, args)
    return collection


def filter_collection(
    collection: ee.ImageCollection,
    coords: list[float],
    period: tuple[str, str, str, str],
) -> ee.ImageCollection:
    filtered = collection
    if period is not None:
        # filtered = filtered.filterDate(*period)  # filter time, if there's one period
        filtered = filtered.filter(
            ee.Filter.Or(
                ee.Filter.date(period[0], period[1]),
                ee.Filter.date(period[2], period[3]),
            )
        )  # filter time, if there're two periods

    filtered = filtered.filterBounds(ee.Geometry.Point(coords))  # filter region

    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f"ImageCollection.filter: No suitable images found in ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}."  # noqa: E501
        )
    return filtered


def center_crop(
    img: np.ndarray[Any, np.dtype[Any]], out_size: tuple[int, int]
) -> np.ndarray[Any, np.dtype[Any]]:
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = (image_height - crop_height + 1) // 2
    crop_left = (image_width - crop_width + 1) // 2
    return img[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]


def adjust_coords(
    coords: list[list[float]], old_size: tuple[int, int], new_size: tuple[int, int]
) -> list[list[float]]:
    xres = (coords[1][0] - coords[0][0]) / old_size[1]
    yres = (coords[0][1] - coords[1][1]) / old_size[0]
    xoff = int((old_size[1] - new_size[1] + 1) * 0.5)
    yoff = int((old_size[0] - new_size[0] + 1) * 0.5)
    return [
        [coords[0][0] + (xoff * xres), coords[0][1] - (yoff * yres)],
        [
            coords[0][0] + ((xoff + new_size[1]) * xres),
            coords[0][1] - ((yoff + new_size[0]) * yres),
        ],
    ]


def get_properties(image: ee.Image) -> Any:
    return image.getInfo()


def get_patch(
    collection: ee.ImageCollection,
    center_coord: list[float],
    radius: float,
    bands: list[str],
    crop: Optional[dict[str, Any]] = None,
    dtype: str = "float32",
) -> dict[str, Any]:
    image = collection.sort("system:time_start", False).first()  # get most recent
    region = (
        ee.Geometry.Point(center_coord).buffer(radius).bounds()
    )  # sample region bound
    reproj_bands = "B8"
    reproj_resolution = 30
    other_bands = [b for b in bands if b != reproj_bands]
    patch_reproj = (
        image.select(reproj_bands)
        .reproject(
            crs=image.select(reproj_bands).projection().crs(), scale=reproj_resolution
        )
        .sampleRectangle(region, defaultValue=0)
    )
    patch_other = image.select(*other_bands).sampleRectangle(region, defaultValue=0)
    # patch = image.select(*bands).sampleRectangle(region, defaultValue=0)
    # features = patch.getInfo()  # the actual download
    features_reproj = patch_reproj.getInfo()
    features_other = patch_other.getInfo()

    raster = OrderedDict()
    for band in bands:
        if band == reproj_bands:
            img = np.atleast_3d(features_reproj["properties"][band])
        else:
            img = np.atleast_3d(features_other["properties"][band])
        if crop is not None:
            img = center_crop(img, out_size=crop[band])
        raster[band] = img.astype(dtype)

    coords0 = np.array(features_other["geometry"]["coordinates"][0])
    coords = [
        [coords0[:, 0].min(), coords0[:, 1].max()],
        [coords0[:, 0].max(), coords0[:, 1].min()],
    ]
    if crop is not None:
        band = bands[0]
        old_size = (
            len(features_other["properties"][band]),
            len(features_other["properties"][band][0]),
        )
        new_size = raster[band].shape[:2]
        coords = adjust_coords(coords, old_size, new_size)

    return OrderedDict(
        {"raster": raster, "coords": coords, "metadata": get_properties(image)}
    )


def get_random_patches_match(
    idx: int,
    collection: ee.ImageCollection,
    bands: list[str],
    crops: dict[str, Any],
    dtype: str,
    dates: list[Any],
    radius: float,
    debug: bool = False,
    match_coords: dict[str, Any] = {},
) -> tuple[Optional[list[dict[str, Any]]], list[float]]:
    # (lon,lat) of idx patch
    coords = match_coords[str(idx)]

    # random +- 30 days of random days within 1 year from the reference dates
    periods = [get_period(date, days=60) for date in dates]

    try:
        filtered_collections = [
            filter_collection(collection, coords, p) for p in periods
        ]
        patches = [
            get_patch(c, coords, radius, bands=bands, crop=crops, dtype=dtype)
            for c in filtered_collections
        ]

    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        return None, coords

    return patches, coords


def save_geotiff(
    img: np.ndarray[Any, np.dtype[Any]], coords: list[list[float]], filename: str
) -> None:
    height, width, channels = img.shape
    xres = (coords[1][0] - coords[0][0]) / width
    yres = (coords[0][1] - coords[1][1]) / height
    transform = Affine.translation(
        coords[0][0] - xres / 2, coords[0][1] + yres / 2
    ) * Affine.scale(xres, -yres)
    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": channels,
        "crs": "+proj=latlong",
        "transform": transform,
        "dtype": img.dtype,
        "compress": "None",
    }
    with rasterio.open(filename, "w", **profile) as f:
        f.write(img.transpose(2, 0, 1))


def save_patch(
    raster: dict[str, Any],
    coords: list[list[float]],
    metadata: dict[str, Any],
    path: str,
) -> None:
    patch_id = metadata["properties"]["system:index"]
    patch_path = os.path.join(path, patch_id)
    os.makedirs(patch_path, exist_ok=True)

    img_all = np.concatenate([img for img in raster.values()], axis=2)
    save_geotiff(img_all, coords, os.path.join(patch_path, "all_bands.tif"))
    # for band, img in raster.items():
    #    save_geotiff(img, coords, os.path.join(patch_path, f"{band}.tif"))

    with open(os.path.join(patch_path, "metadata.json"), "w") as f:
        json.dump(metadata, f)


class Counter:
    def __init__(self, start: int = 0) -> None:
        self.value = start
        self.lock = Lock()

    def update(self, delta: int = 1) -> int:
        with self.lock:
            self.value += delta
            return self.value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", type=str, default="./data/", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection", type=str, default="COPERNICUS/S2", help="GEE collection name"
    )
    parser.add_argument(
        "--qa-band", type=str, default="QA60", help="qa band name (optional)"
    )  # optional
    parser.add_argument(
        "--qa-cloud-bit", type=int, default=10, help="qa band cloud bit (optional)"
    )  # optional
    parser.add_argument(
        "--meta-cloud-name",
        type=str,
        default="CLOUDY_PIXEL_PERCENTAGE",
        help="meta data cloud percentage name",
    )
    parser.add_argument(
        "--cloud-pct", type=int, default=20, help="cloud percentage threshold"
    )
    # patch properties
    parser.add_argument(
        "--dates",
        type=str,
        nargs="+",
        default=["2021-12-21", "2021-09-22", "2021-06-21", "2021-03-20"],
        help="reference dates",
    )
    parser.add_argument(
        "--radius", type=int, default=1320, help="patch radius in meters"
    )
    parser.add_argument(
        "--bands",
        type=str,
        nargs="+",
        default=[
            "B1",
            "B2",
            "B3",
            "B4",
            "B5",
            "B6",
            "B7",
            "B8",
            "B8A",
            "B9",
            "B10",
            "B11",
            "B12",
        ],
        help="bands to download",
    )
    parser.add_argument(
        "--crops",
        type=int,
        nargs="+",
        default=[44, 264, 264, 264, 132, 132, 132, 264, 132, 44, 44, 132, 132],
        help="crop size for each band",
    )
    parser.add_argument("--dtype", type=str, default="float32", help="data type")
    # download settings
    parser.add_argument("--num-workers", type=int, default=8, help="number of workers")
    parser.add_argument("--log-freq", type=int, default=10, help="print frequency")
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a previous run"
    )
    # sampler options
    parser.add_argument(
        "--match-file",
        type=str,
        required=True,
        help="match pre-sampled coordinates and indexes",
    )
    # number of locations to download
    parser.add_argument(
        "--indices-range",
        type=int,
        nargs=2,
        default=[0, 250000],
        help="indices to download",
    )
    # debug
    parser.add_argument("--debug", action="store_true", help="debug mode")

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # initialize ee
    ee.Initialize()

    # get data collection (remove clouds)
    collection = get_collection(args.collection, args.meta_cloud_name, args.cloud_pct)

    dates = []
    for d in args.dates:
        dates.append(date.fromisoformat(d))

    bands = args.bands
    crops = {}
    for i, band in enumerate(bands):
        crops[band] = (args.crops[i], args.crops[i])
    dtype = args.dtype

    # if resume
    ext_coords = {}
    ext_flags = {}
    if args.resume:
        ext_path = args.resume
        with open(ext_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2)  # lon, lat
                ext_flags[key] = int(row[3])  # success or not
    else:
        ext_path = os.path.join(args.save_path, "checked_locations.csv")

    # match from pre-sampled coords
    match_coords = {}
    with open(args.match_file) as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            key = row[0]
            val1 = float(row[1])
            val2 = float(row[2])
            match_coords[key] = (val1, val2)  # lon, lat

    start_time = time.time()
    counter = Counter()

    def worker(idx: int) -> None:
        if str(idx) in ext_coords.keys():
            return

        patches, center_coord = get_random_patches_match(
            idx,
            collection,
            bands,
            crops,
            dtype,
            dates,
            radius=args.radius,
            debug=args.debug,
            match_coords=match_coords,
        )

        if patches is not None:
            if args.save_path is not None:
                location_path = os.path.join(args.save_path, "imgs", f"{idx:07d}")
                os.makedirs(location_path, exist_ok=True)
                for patch in patches:
                    save_patch(
                        raster=patch["raster"],
                        coords=patch["coords"],
                        metadata=patch["metadata"],
                        path=location_path,
                    )

            count = counter.update(1)
            if count % args.log_freq == 0:
                print(f"Downloaded {count} images in {time.time() - start_time:.3f}s.")
        else:
            print("no suitable image for location %d." % (idx))

        # add to existing checked locations
        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            if patches is not None:
                success = 1
            else:
                success = 0
            data = [idx, *center_coord, success]
            writer.writerow(data)

        return

    # set indices
    if args.indices_range is not None:
        indices = list(range(args.indices_range[0], args.indices_range[1]))
    else:
        indices = []
        for key in match_coords.keys():
            indices.append(int(key))

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        # parallelism data
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
