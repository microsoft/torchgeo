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
    --resume \
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

import numpy as np
import pandas as pd
from rtree import index
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def get_world_cities(
    download_root: str = "world_cities", size: int = 10000
) -> pd.DataFrame:
    url = "https://simplemaps.com/static/data/world-cities/basic/simplemaps_worldcities_basicv1.71.zip"  # noqa: E501
    filename = "worldcities.csv"
    download_and_extract_archive(url, download_root)
    cols = ["city", "lat", "lng", "population"]
    cities = pd.read_csv(os.path.join(download_root, filename), usecols=cols)
    cities.at[8436, "population"] = 50789  # fix one bug (Tecax) in the csv file
    cities = cities.nlargest(size, "population")
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
        "--resume", action="store_true", help="resume from a previous run"
    )
    parser.add_argument(
        "--indices-range",
        type=int,
        nargs=2,
        default=[0, 250000],
        help="indices to sample",
    )

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    path = os.path.join(args.save_path, "sampled_locations.csv")
    root = os.path.join(args.save_path, "world_cities")
    cities = get_world_cities(download_root=root, size=args.num_cities)
    bbox_size = args.size / 1000  # no overlap between adjacent patches
    bbox_size_degree = km2deg(bbox_size)

    # Populate R-tree if resuming
    rtree_coords = index.Index()
    if args.resume:
        print("Loading existing locations...")
        with open(path) as csv_file:
            reader = csv.reader(csv_file)
            for i, row in enumerate(reader):
                key = int(row[0])
                val1 = float(row[1])
                val2 = float(row[2])
                bbox = create_bbox((val1, val2), bbox_size_degree)
                rtree_coords.insert(i, bbox)
        assert key < args.indices_range[0]
    else:
        if os.path.exists(path):
            os.remove(path)

    # Sample locations and save to file
    print("Sampling new locations...")
    start_time = time.time()
    with open(path, "a") as f:
        writer = csv.writer(f)
        for i in tqdm(range(*args.indices_range)):
            # Sample new coord and check overlap
            while True:
                # (lon,lat) of top-10000 cities
                new_coord = sample_point(cities, args.std)
                bbox = create_bbox(new_coord, bbox_size_degree)
                if not list(rtree_coords.intersection(bbox)):
                    break

            rtree_coords.insert(i, bbox)
            data = [i, *new_coord]
            writer.writerow(data)
            f.flush()

    elapsed = time.time() - start_time
    print(f"Sampled locations saved to {path} in {elapsed:.2f} seconds.")
