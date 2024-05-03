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
from shapely.geometry import MultiPolygon, Point, shape
from shapely.ops import unary_union
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def retrieve_rois_polygons(download_root: str) -> MultiPolygon:
    """Retrieve CONUS MultiPolygon.

    Args:
        download_root: directory where to store usa shape file

    Returns:
        MultiPolygon of CONUS
    """
    state_url = 'https://www2.census.gov/geo/tiger/GENZ2018/shp/cb_2018_us_state_5m.zip'
    state_filename = 'cb_2018_us_state_5m.shp'
    download_and_extract_archive(state_url, download_root)

    excluded_states = [
        'United States Virgin Islands',
        'Commonwealth of the Northern Mariana Islands',
        'Puerto Rico',
        'Alaska',
        'Hawaii',
        'American Samoa',
        'Guam',
    ]
    conus = []
    with fiona.open(os.path.join(download_root, state_filename), 'r') as shapefile:
        for feature in shapefile:
            name = feature['properties']['NAME']
            if name in excluded_states:
                continue
            else:
                conus.append(shape(feature['geometry']))

    conus = unary_union(conus)
    return conus


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--save-path', type=str, default='./data/', help='dir to save data'
    )
    parser.add_argument(
        '--size', type=float, default=1320, help='half patch size in meters'
    )
    parser.add_argument(
        '--indices-range',
        type=int,
        nargs=2,
        default=[0, 500],
        help='indices to download',
    )
    parser.add_argument(
        '--resume', action='store_true', help='resume from a previous run'
    )

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    bbox_size = args.size / 1000  # no overlap between adjacent patches
    bbox_size_degree = km2deg(bbox_size)

    root = os.path.join(args.save_path, 'conus')
    csv_path = os.path.join(args.save_path, 'sampled_locations.csv')

    # Populate R-tree if resuming
    rtree_coords = index.Index()
    if args.resume:
        print('Loading existing locations...')
        with open(csv_path) as csv_file:
            reader = csv.reader(csv_file)
            for i, row in enumerate(reader):
                key = int(row[0])
                val1 = float(row[1])
                val2 = float(row[2])
                bbox = create_bbox((val1, val2), bbox_size_degree)
                rtree_coords.insert(i, bbox)
        assert key < args.indices_range[0]
    else:
        if os.path.exists(csv_path):
            os.remove(csv_path)

    # Retrieve Area of interest and states to ignore
    conus = retrieve_rois_polygons(root)
    x_min, y_min, x_max, y_max = conus.bounds

    with open(csv_path, 'a') as f:
        writer = csv.writer(f)
        for idx in tqdm(range(*args.indices_range)):
            count = 0
            while count < 1:
                x = random.uniform(x_min, x_max)
                y = random.uniform(y_min, y_max)
                point = Point(x, y)
                if conus.contains(point):
                    bbox = create_bbox((x, y), bbox_size_degree)
                    if list(rtree_coords.intersection(bbox)):
                        continue
                    else:
                        rtree_coords.insert(idx, bbox)
                        count += 1
                        # write to file
                        data = [idx, x, y]
                        writer.writerow(data)
