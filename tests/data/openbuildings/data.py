#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import csv
import gzip
import hashlib
import json
import os
import shutil

from shapely.geometry import Polygon

SIZE = 0.05


def create_meta_data_file(zipfilename):
    meta_data = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [[0.0, 0.0], [0.0, SIZE], [SIZE, SIZE], [SIZE, 0.0], [0.0, 0.0]]
                    ],
                },
                "properties": {
                    "tile_id": "025",
                    "tile_url": f"polygons_s2_level_4_gzip/{zipfilename}",
                    "size_mb": 0.2,
                },
            }
        ],
    }
    return meta_data


def create_csv_data_row(lat, long):
    width, height = SIZE / 10, SIZE / 10
    minx = long - 0.5 * width
    maxx = long + 0.5 * width
    miny = lat - 0.5 * height
    maxy = lat - 0.5 * height
    coordinates = [(minx, miny), (minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)]
    polygon = Polygon(coordinates)

    data_row = {
        "latitude": lat,
        "longitude": long,
        "area_in_meters": 1.0,
        "confidence": 1.0,
        "geometry": polygon.wkt,
        "full_plus_code": "ABC",
    }

    return data_row


def create_buildings_data():
    fourth = SIZE / 4
    # pandas df
    dict_data = [
        create_csv_data_row(fourth, fourth),
        create_csv_data_row(SIZE - fourth, SIZE - fourth),
    ]
    return dict_data


if __name__ == "__main__":
    csvname = "000_buildings.csv"
    zipfilename = csvname + ".gz"

    # create and save metadata
    meta_data = create_meta_data_file(zipfilename)
    with open("tiles.geojson", "w") as fp:
        json.dump(meta_data, fp)

    # create and archive buildings data
    buildings_data = create_buildings_data()
    keys = buildings_data[0].keys()
    with open(csvname, "w") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        w.writerows(buildings_data)

    # archive the csv to gzip
    with open(csvname, "rb") as f_in:
        with gzip.open(zipfilename, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

    # Compute checksums
    with open(zipfilename, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f"{zipfilename}: {md5}")

    # remove csv file
    os.remove(csvname)
