#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os

import ee
import numpy as np
from geeS2downloader import GEES2Downloader


def download_data(args: argparse.Namespace) -> None:
    """Download Data from GEE.

    Args:
        args: argparse namespace
    """
    # initialize seed
    os.makedirs(args.save_path, exist_ok=True)

    # initialize ee
    ee.Initialize()

    downloader = GEES2Downloader()

    conus = ee.FeatureCollection("TIGER/2018/States")  # this takes ages
    conus = ee.Geometry.Polygon(
        [
            [-127.86, 50.18],
            [-65.39, 50.18],
            [-65.39, 23.67],
            [-127.86, 23.67],
            [-127.86, 50.18],
        ]
    )

    # get data collection and filter bounds, date, and cloud
    num_samples = 100
    collection = (
        ee.ImageCollection(args.collection)
        .filterBounds(conus)  # adding this filter takes ages
        .filterDate(args.start_date, args.end_date)
        .filter(
            ee.Filter.And(
                ee.Filter.gte(args.meta_cloud_name, 0),
                ee.Filter.lte(args.meta_cloud_name, args.cloud_pct),
            )
        )
    )

    random_collection = collection.randomColumn().sort("random").limit(num_samples)
    listOfImages = random_collection.toList(random_collection.size())

    # Download the images
    for i in range(num_samples):
        image = ee.Image(listOfImages.get(i))

        collected_bands = []

        for band in args.bands:
            downloader.download(image, band, scale=30)
            collected_bands.append(downloader.array)

        array = np.stack(collected_bands, axis=-1)

        # save to geotiff with meta info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-path", type=str, default="./data/", help="dir to save data"
    )
    # collection properties
    parser.add_argument(
        "--collection",
        type=str,
        default="LANDSAT/LC08/C02/T1_TOA",
        help="GEE collection name",
    )
    # clouds
    parser.add_argument(
        "--meta-cloud-name",
        type=str,
        default="CLOUD_COVER",
        help="meta data cloud percentage name",
    )
    parser.add_argument(
        "--cloud-pct", type=int, default=20, help="cloud percentage threshold"
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
            "B9",
            "B10",
            "B11",
            "B12",
        ],
        help="bands to download",
    )
    # tile properties
    parser.add_argument(
        "--start_date", type=str, default="2019-05-01", help="start date"
    )
    parser.add_argument("--end_date", type=str, default="2019-08-31", help="end date")

    args = parser.parse_args()
    download_data(args)
