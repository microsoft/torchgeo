#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import csv
import os
import random

import fiona
from rtree import index
from sample_ssl4eo import create_bbox, km2deg
from shapely.geometry import Point, shape
from torchvision.datasets.utils import download_and_extract_archive


def get_uniform_points_within_conus(
    download_root: str, num_samples: int, bbox_size_degree: float
) -> list[tuple[float, float]]:
    random.seed(0)
    nation_url = (
        "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_nation_5m.zip"
    )
    nation_filename = "cb_2022_us_nation_5m.shp"
    download_and_extract_archive(nation_url, download_root)

    state_url = "https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_5m.zip"
    state_filename = "cb_2018_us_state_5m.shp"
    download_and_extract_archive(state_url, download_root)

    exclude_states = [
        "United States Virgin Islands",
        "Commonwealth of the Northern Mariana Island",
        "Puerto Rico",
        "Alaska",
        "Hawaii",
        "American Samoa",
        "Guam",
    ]
    with fiona.open(os.path.join(download_root, state_filename), "r") as shapefile:
        excluded = []
        for feature in shapefile:
            name = feature["properties"]["NAME"]
            if name in exclude_states:
                excluded.append(shape(feature["geometry"]))

    rtree_coords = index.Index()
    with fiona.open(os.path.join(download_root, nation_filename), "r") as shapefile:
        conus = shape(shapefile[0]["geometry"])
        x_min, y_min, x_max, y_max = conus.bounds
        points: list[tuple[float, float]] = []
        while len(points) < num_samples:
            x = random.uniform(x_min, x_max)
            y = random.uniform(y_min, y_max)
            point = Point(x, y)
            if conus.contains(point) and not any(
                [polygon.contains(point) for polygon in excluded]
            ):
                bbox = create_bbox((x, y), bbox_size_degree)
                if list(rtree_coords.intersection(bbox)):
                    continue
                else:
                    rtree_coords.insert(len(points), bbox)
                    points.append((x, y))

    return points


def save_csv(points: list[tuple[float, float]], ext_path: str) -> None:
    with open(ext_path, "w") as f:
        writer = csv.writer(f)
        for idx, (lng, lat) in enumerate(points):
            data = [idx, lng, lat]
            writer.writerow(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", type=str, default="./data/", help="dir to save data"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="number of uniform samples to draw from conus",
    )
    parser.add_argument(
        "--size", type=float, default=1320, help="half patch size in meters"
    )

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    bbox_size = args.size / 1000  # no overlap between adjacent patches
    bbox_size_degree = km2deg(bbox_size)

    root = os.path.join(args.save_path, "conus")
    points = get_uniform_points_within_conus(root, args.num_samples, bbox_size_degree)
    save_csv(points, os.path.join(root, "sampled_locations.csv"))
