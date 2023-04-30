#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

""" Sample patch locations for downloading

### run the script:
## sample new locations with rtree overlap search
python sample_ssl4eo.py \
    --save-path ./data \
    --size 1320 \
    --num-cities 10000 \
    --std 50 \
    --indices-range 0 250000

## resume from interruption
python sample_ssl4eo.py \
    --save-path ./data \
    --size 1320 \
    --num-cities 10000 \
    --std 50 \
    --resume ./data/sampled_locations.csv \
    --indices-range 0 250000

### Notes
# The script will sample locations with rtree overlap search.
# The script will save the sampled locations to a csv file.
# By default, GaussianSampler is used to sample locations with a standard deviation
 of 50 km from top 10000 populated cities.
# Size (meter) is half the wanted patch size.

"""

import argparse
import csv
import os
import time
import warnings

import numpy as np
import pandas as pd
from rtree import index
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm

warnings.simplefilter("ignore", UserWarning)


""" samplers to get locations of interest points"""


def get_world_cities(
    download_root: str = "world_cities", size: int = 10000
) -> pd.DataFrame:
    url = "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.71.zip"  # noqa: E501
    filename = "worldcities.csv"
    if not os.path.exists(os.path.join(download_root, os.path.basename(url))):
        download_and_extract_archive(url, download_root)
    cols = ["city", "lat", "lng", "population"]
    cities = pd.read_csv(os.path.join(download_root, filename), usecols=cols)
    cities.at[8436, "population"] = 50789  # fix one bug (Tecax) in the csv file
    cities = cities.sort_values(by=["population"], ascending=False).head(size)
    return cities


def km2deg(kms: float, radius: float = 6371) -> float:
    return kms / (2.0 * radius * np.pi / 360.0)


def sample_point(cities: pd.DataFrame, std: float) -> tuple[float, float]:
    city = cities.sample()
    point = (float(city["lng"]), float(city["lat"]))
    std = km2deg(std)
    lon, lat = np.random.normal(loc=point, scale=[std, std])
    return (lon, lat)


def create_bbox(
    coords: tuple[float, float], bbox_size_degree: float
) -> tuple[float, float, float, float]:
    lon, lat = coords
    bbox = (
        lon - bbox_size_degree,
        lat - bbox_size_degree,
        lon + bbox_size_degree,
        lat + bbox_size_degree,
    )
    return bbox


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", type=str, default="./data/", help="dir to save data"
    )
    parser.add_argument(
        "--size", type=float, default=1320, help="half patch size in meters"
    )
    parser.add_argument(
        "--num-cities", type=int, default=10000, help="number of cities to sample"
    )
    parser.add_argument(
        "--std", type=int, default=50, help="std dev of gaussian distribution"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="resume from a previous run"
    )
    parser.add_argument(
        "--indices-range",
        type=int,
        nargs=2,
        default=[0, 250000],
        help="indices to download",
    )

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    # if resume
    ext_coords = {}
    if args.resume:
        ext_path = args.resume
        with open(ext_path) as csv_file:
            reader = csv.reader(csv_file)
            for row in reader:
                key = row[0]
                val1 = float(row[1])
                val2 = float(row[2])
                ext_coords[key] = (val1, val2)  # lon, lat
    else:
        ext_path = os.path.join(args.save_path, "sampled_locations.csv")

    # initialize sampler
    root = os.path.join(args.save_path, "world_cities")
    cities = get_world_cities(download_root=root, size=args.num_cities)
    bbox_size = args.size / 1000  # no overlap between adjacent patches
    bbox_size_degree = km2deg(bbox_size)

    # build rtree
    rtree_coords = index.Index()
    if args.resume:
        print("Load existing locations.")
        for i, key in enumerate(tqdm(ext_coords.keys())):
            c = ext_coords[key]
            bbox = create_bbox(c, bbox_size_degree)
            rtree_coords.insert(i, bbox)

    # sample locations
    start_time = time.time()
    indices = range(*args.indices_range)
    new_coords = {}
    for idx in tqdm(indices):
        # skip if already sampled
        if str(idx) in ext_coords.keys():
            if args.resume:
                continue

        # sample new coord and check overlap
        count = 0
        while count < 1:
            new_coord = sample_point(cities, args.std)  # (lon,lat) of top-10000 cities
            bbox = create_bbox(new_coord, bbox_size_degree)
            if list(rtree_coords.intersection(bbox)):
                continue
            rtree_coords.insert(idx, bbox)
            new_coords[idx] = new_coord
            count += 1

    # save to file
    with open(ext_path, "a") as f:
        writer = csv.writer(f)
        for idx, new_coord in new_coords.items():
            data = [idx, *new_coord]
            writer.writerow(data)

    print(
        f"Sampled locations saved to {ext_path} in {time.time()-start_time:.2f} seconds."  # noqa: E501
    )
