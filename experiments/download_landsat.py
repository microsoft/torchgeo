#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import numpy as np
import os
import ee

from geeS2downloader import GEES2Downloader

def download_data(args):
    
    # initialize seed
    os.makedirs(args.save_path, exist_ok=True)

    # initialize ee
    ee.Initialize()

    downloader = GEES2Downloader()

    conus = ee.FeatureCollection("TIGER/2018/States")

    # get data collection (remove clouds)
    num_samples = 100
    collection = (
        ee.ImageCollection(args.collection)
        .filterBounds(conus)
        .filterDate(args.start_date, args.end_date)
        .filterMetadata("CLOUD COVER", "less_than", str(args.cloud_pct))
        
    )
    random_collection = collection.randomColumn().sort('random').limit(num_samples)
    listOfImages = random_collection.toList(random_collection.size())

    # Download the images
    for i in range(num_samples):
        image = ee.Image(listOfImages.get(i))
        # url = image.getDownloadURL(
        #     {"name": "landsat_random_image_" + str(i), "scale": 30, "crs": "EPSG:4326"}
        # )
    
        collected_bands = []
        band_names = image.bandNames()
        for band in band_names:
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
        default="LANDSAT/LC08/C02/T1_L2",
        help="GEE collection name",
    )
    # clouds
    parser.add_argument(
        "--meta-cloud-name",
        type=str,
        default="CLOUDY_PIXEL_PERCENTAGE",
        help="meta data cloud percentage name",
    )
    parser.add_argument(
        "--cloud-pct", type=int, default=20, help="cloud percentage threshold"
    )
    # tile properties
    parser.add_argument(
        "--start_date", type=str, default="2019-05-01", help="start date"
    )
    parser.add_argument("--end_date", type=str, default="2019-08-31", help="end date")

    args = parser.parse_args()
    download_data(args)
