#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Sample and download Sentinel-1/2 images with Google Earth Engine

#### run the script:
### match and download pre-sampled locations
python download_ssl4eo.py \
    --save-path ./data \
    --num-workers 8 \
    --cloud-pct 20 \
    --log-freq 100 \
    --match-file ./data/sampled_locations.csv \
    --indices-range 0 250000

### resume from interruption
python download_ssl4eo.py \
    -- ... \
    --resume ./data/checked_locations.csv \
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
from typing import Any, Dict, List, Optional, Tuple

import ee
import numpy as np
import rasterio
import urllib3
from rasterio.transform import Affine

warnings.simplefilter("ignore", UserWarning)

ALL_BANDS_L2A = [
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
    "B11",
    "B12",
]
ALL_BANDS_L1C = [
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
]
RGB_BANDS = ["B4", "B3", "B2"]
ALL_BANDS_GRD = ["VV", "VH"]


def date2str(date: datetime) -> str:
    return date.strftime("%Y-%m-%d")


def get_period(date: datetime, days: int = 5) -> Tuple[str, str]:
    date1 = date - timedelta(days=days / 2)
    date2 = date + timedelta(days=days / 2)
    return date2str(date1), date2str(date2)


"""get collection and remove clouds from ee"""


def maskS2clouds(image: ee.Image) -> ee.Image:
    qa = image.select("QA60")
    # Bits 10 and 11 are clouds and cirrus, respectively.
    cloudBitMask = 1 << 10
    cirrusBitMask = 1 << 11
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudBitMask).eq(0)
    mask = mask.bitwiseAnd(cirrusBitMask).eq(0)
    return image.updateMask(mask)


def get_collection_s2a(cloud_pct: float = 20) -> ee.ImageCollection:
    collection = ee.ImageCollection("COPERNICUS/S2_SR")
    collection = collection.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    # Uncomment the following line if you want to apply cloud masking.
    # collection = collection.map(maskS2clouds)
    return collection


def get_collection_s2c(cloud_pct: float = 20) -> ee.ImageCollection:
    collection = ee.ImageCollection("COPERNICUS/S2")
    collection = collection.filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
    # Uncomment the following line if you want to apply cloud masking.
    # collection = collection.map(maskS2clouds)
    return collection


def get_collection_s1() -> ee.ImageCollection:
    collection = ee.ImageCollection("COPERNICUS/S1_GRD")
    return collection


def filter_collection(
    collection: ee.ImageCollection, coords: List[float], period: Tuple[str, str]
) -> ee.ImageCollection:
    filtered = collection
    if period is not None:
        filtered = filtered.filterDate(*period)  # filter time
    filtered = filtered.filterBounds(ee.Geometry.Point(coords))  # filter region

    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            f"ImageCollection.filter: No suitable images found in \
                ({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}."
        )
    return filtered


def filter_collection_s1(
    collection: ee.ImageCollection, coords: List[float], period: Tuple[str, str]
) -> ee.ImageCollection:
    filtered = collection
    if period is not None:
        filtered = filtered.filterDate(*period)  # filter time
    filtered = filtered.filterBounds(ee.Geometry.Point(coords))  # filter region

    filtered = filtered.filter(
        ee.Filter.listContains("transmitterReceiverPolarisation", "VV")
    ).filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VH"))
    filtered = filtered.filter(ee.Filter.eq("instrumentMode", "IW"))
    # If you want to filter by orbit properties, uncomment the following line.
    # filtered = filtered.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
    # filtered = filtered.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'))

    if filtered.size().getInfo() == 0:
        raise ee.EEException(
            "ImageCollection.filter: No suitable images found in "
            f"({coords[1]:.4f}, {coords[0]:.4f}) between {period[0]} and {period[1]}."
        )
    return filtered


def center_crop(
    img: np.ndarray[Any, np.dtype[Any]], out_size: Tuple[int, int]
) -> np.ndarray[Any, np.dtype[Any]]:
    image_height, image_width = img.shape[:2]
    crop_height, crop_width = out_size
    crop_top = (image_height - crop_height + 1) // 2
    crop_left = (image_width - crop_width + 1) // 2
    return img[crop_top : crop_top + crop_height, crop_left : crop_left + crop_width]


def adjust_coords(
    coords: List[List[float]], old_size: Tuple[int, int], new_size: Tuple[int, int]
) -> List[List[float]]:
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


def get_patch_s1(
    collection: ee.ImageCollection,
    center_coord: List[float],
    radius: float,
    bands: List[str],
    crop: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    image = collection.sort("system:time_start", False).first()  # get most recent
    region = (
        ee.Geometry.Point(center_coord).buffer(radius).bounds()
    )  # sample region bound

    patch = image.select(*bands).sampleRectangle(region, defaultValue=0)
    features = patch.getInfo()  # the actual download

    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(features["properties"][band])
        if crop is not None:
            img = center_crop(img, out_size=crop[band])
        raster[band] = img.astype("float32")

    coords0 = np.array(features["geometry"]["coordinates"][0])
    coords = [
        [coords0[:, 0].min(), coords0[:, 1].max()],
        [coords0[:, 0].max(), coords0[:, 1].min()],
    ]
    if crop is not None:
        band = bands[0]
        old_size = (
            len(features["properties"][band]),
            len(features["properties"][band][0]),
        )
        new_size = raster[band].shape[:2]
        coords = adjust_coords(coords, old_size, new_size)

    return OrderedDict(
        {"raster": raster, "coords": coords, "metadata": get_properties(image)}
    )


def get_patch_s2(
    collection: ee.ImageCollection,
    center_coord: List[float],
    radius: float,
    bands: Optional[List[str]] = None,
    crop: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if bands is None:
        bands = RGB_BANDS

    image = collection.sort("system:time_start", False).first()  # get most recent
    region = (
        ee.Geometry.Point(center_coord).buffer(radius).bounds()
    )  # sample region bound
    patch = image.select(*bands).sampleRectangle(region, defaultValue=0)

    features = patch.getInfo()  # the actual download

    raster = OrderedDict()
    for band in bands:
        img = np.atleast_3d(features["properties"][band])
        if crop is not None:
            img = center_crop(img, out_size=crop[band])
        raster[band] = img.astype("uint16")

    coords0 = np.array(features["geometry"]["coordinates"][0])
    coords = [
        [coords0[:, 0].min(), coords0[:, 1].max()],
        [coords0[:, 0].max(), coords0[:, 1].min()],
    ]
    if crop is not None:
        band = bands[0]
        old_size = (
            len(features["properties"][band]),
            len(features["properties"][band][0]),
        )
        new_size = raster[band].shape[:2]
        coords = adjust_coords(coords, old_size, new_size)

    return OrderedDict(
        {"raster": raster, "coords": coords, "metadata": get_properties(image)}
    )


def get_random_patches_match(
    idx: int,
    collections: Dict[str, Any],
    bands: Dict[str, Any],
    crops: Dict[str, Any],
    dates: List[Any],
    radius: int,
    debug: bool = False,
    match_coords: Dict[str, Any] = {},
) -> Tuple[
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    Optional[List[Dict[str, Any]]],
    List[float],
]:
    # (lon,lat) of idx patch
    coords = match_coords[str(idx)]

    # random +- 15 days of random days within 1 year from the reference dates
    delta = timedelta(days=np.random.randint(365))
    periods = [get_period(date - delta, days=30) for date in dates]

    collection_s1 = collections["s1_grd"]
    collection_s2c = collections["s2_l1c"]
    collection_s2a = collections["s2_l2a"]

    bands_s1 = bands["s1_grd"]
    bands_s2c = bands["s2_l1c"]
    bands_s2a = bands["s2_l2a"]

    crop_s1 = crops["s1_grd"]
    crop_s2c = crops["s2_l1c"]
    crop_s2a = crops["s2_l2a"]

    try:
        filtered_collections_s2c = [
            filter_collection(collection_s2c, coords, p) for p in periods
        ]
        patches_s2c = [
            get_patch_s2(c, coords, radius, bands=bands_s2c, crop=crop_s2c)
            for c in filtered_collections_s2c
        ]
        filtered_collections_s2a = [
            filter_collection(collection_s2a, coords, p) for p in periods
        ]
        patches_s2a = [
            get_patch_s2(c, coords, radius, bands=bands_s2a, crop=crop_s2a)
            for c in filtered_collections_s2a
        ]
        filtered_collections_s1 = [
            filter_collection_s1(collection_s1, coords, p) for p in periods
        ]
        patches_s1 = [
            get_patch_s1(c, coords, radius, bands=bands_s1, crop=crop_s1)
            for c in filtered_collections_s1
        ]

        center_coord = coords

    except (ee.EEException, urllib3.exceptions.HTTPError) as e:
        if debug:
            print(e)
        return None, None, None, coords

    return patches_s1, patches_s2c, patches_s2a, center_coord


def save_geotiff(
    img: np.ndarray[Any, np.dtype[Any]], coords: List[List[float]], filename: str
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
    raster: Dict[str, Any],
    coords: List[List[float]],
    metadata: Dict[str, Any],
    path: str,
) -> None:
    patch_id = metadata["properties"]["system:index"]
    patch_path = os.path.join(path, patch_id)
    os.makedirs(patch_path, exist_ok=True)

    for band, img in raster.items():
        save_geotiff(img, coords, os.path.join(patch_path, f"{band}.tif"))

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


def fix_random_seeds(seed: int = 42) -> None:
    """
    Fix random seeds.
    """
    np.random.seed(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-path", type=str, default="./data/")  # dir to save data
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cloud-pct", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-freq", type=int, default=10)  # print frequency
    parser.add_argument(
        "--resume", type=str, default=None
    )  # resume from existing coordinates
    parser.add_argument(
        "--match-file",
        type=str,
        required=True,
        help="match pre-sampled coordinates and indexes",
    )
    parser.add_argument(
        "--indices-range", type=int, nargs=2, default=[0, 250000]
    )  # range of download indices --> number of locations
    args = parser.parse_args()

    fix_random_seeds(seed=42)

    # initialize ee
    ee.Initialize()

    # get data collection (remove clouds)
    collection_s2a = get_collection_s2a(cloud_pct=args.cloud_pct)
    collection_s2c = get_collection_s2c(cloud_pct=args.cloud_pct)
    collection_s1 = get_collection_s1()

    collections = {
        "s1_grd": collection_s1,
        "s2_l2a": collection_s2a,
        "s2_l1c": collection_s2c,
    }

    reference = date.fromisoformat("2021-09-22")
    date1 = date.fromisoformat("2021-06-21")
    date2 = date.fromisoformat("2021-03-20")
    date3 = date.fromisoformat("2020-12-21")

    dates = [reference, date1, date2, date3]

    radius = 1320
    crop10 = (264, 264)
    crop20 = (132, 132)
    crop60 = (44, 44)
    # s2 l2a
    crop_s2a = {
        "B1": crop60,
        "B2": crop10,
        "B3": crop10,
        "B4": crop10,
        "B5": crop20,
        "B6": crop20,
        "B7": crop20,
        "B8": crop10,
        "B8A": crop20,
        "B9": crop60,
        "B11": crop20,
        "B12": crop20,
    }
    # s2 l1c
    crop_s2c = {
        "B1": crop60,
        "B2": crop10,
        "B3": crop10,
        "B4": crop10,
        "B5": crop20,
        "B6": crop20,
        "B7": crop20,
        "B8": crop10,
        "B8A": crop20,
        "B9": crop60,
        "B10": crop60,
        "B11": crop20,
        "B12": crop20,
    }
    # s1 grd
    crop_s1 = {"VV": crop10, "VH": crop10}

    crops = {"s1_grd": crop_s1, "s2_l2a": crop_s2a, "s2_l1c": crop_s2c}

    bands = {"s1_grd": ALL_BANDS_GRD, "s2_l2a": ALL_BANDS_L2A, "s2_l1c": ALL_BANDS_L1C}

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

        (patches_s1, patches_s2c, patches_s2a, center_coord) = get_random_patches_match(
            idx,
            collections,
            bands,
            crops,
            dates,
            radius=radius,
            debug=args.debug,
            match_coords=match_coords,
        )

        if (
            patches_s1 is not None
            and patches_s2c is not None
            and patches_s2a is not None
        ):
            if args.save_path is not None:
                # s2c
                location_path_s2c = os.path.join(args.save_path, "s2c", f"{idx:06d}")
                os.makedirs(location_path_s2c, exist_ok=True)
                for patch in patches_s2c:
                    save_patch(
                        raster=patch["raster"],
                        coords=patch["coords"],
                        metadata=patch["metadata"],
                        path=location_path_s2c,
                    )
                # s2a
                location_path_s2a = os.path.join(args.save_path, "s2a", f"{idx:06d}")
                os.makedirs(location_path_s2a, exist_ok=True)
                for patch in patches_s2a:
                    save_patch(
                        raster=patch["raster"],
                        coords=patch["coords"],
                        metadata=patch["metadata"],
                        path=location_path_s2a,
                    )
                # s1
                location_path_s1 = os.path.join(args.save_path, "s1", f"{idx:06d}")
                os.makedirs(location_path_s1, exist_ok=True)
                for patch in patches_s1:
                    save_patch(
                        raster=patch["raster"],
                        coords=patch["coords"],
                        metadata=patch["metadata"],
                        path=location_path_s1,
                    )

            count = counter.update(1)
            if count % args.log_freq == 0:
                print(f"Downloaded {count} images in {time.time() - start_time:.3f}s.")
        else:
            print("no suitable image for location %d." % (idx))

        # add to existing checked locations
        with open(ext_path, "a") as f:
            writer = csv.writer(f)
            if patches_s2c is not None:
                success = 1
            else:
                success = 0
            data = [idx, center_coord[0], center_coord[1], success]
            writer.writerow(data)

        return

    # set indices
    if args.indices_range is not None:
        indices = range(args.indices_range[0], args.indices_range[1])
    else:
        print("Please set up indices.")
        raise NotImplementedError

    if args.num_workers == 0:
        for i in indices:
            worker(i)
    else:
        # parallelism data
        with Pool(processes=args.num_workers) as p:
            p.map(worker, indices)
