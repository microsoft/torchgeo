#!/usr/bin/env python3

# Copyright (c) TorchGeo Contributors. All rights reserved.
# Licensed under the MIT License.

import csv
import zipfile

import geopandas as gpd
from rasterio.crs import CRS
from shapely import Polygon

# Size of example crop field polygon in projection units.
# This is set to align with Sentinel-2 test data, which is a 128x128 image at 10
# projection units per pixel (1280x1280 projection units).
SIZE = 1280


def create_data_file(dataname: str) -> None:
    coordinates = [[0.0, 0.0], [0.0, SIZE], [SIZE, SIZE], [SIZE, 0.0], [0.0, 0.0]]
    # The offset aligns with the Sentinel-2 test data.
    offset = [399960, 4500000 - SIZE]
    polygon = Polygon([[x + offset[0], y + offset[1]] for x, y in coordinates])
    gpd.GeoDataFrame(
        {'EC_hcat_c': ['1000000010']}, geometry=[polygon], crs=CRS.from_epsg(32616)
    ).to_file(dataname, driver='ESRI Shapefile')


def create_csv(fname: str) -> None:
    with open(fname, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['HCAT2_code'])
        writer.writeheader()
        writer.writerow({'HCAT2_code': '1000000000'})
        writer.writerow({'HCAT2_code': '1000000010'})


if __name__ == '__main__':
    csvname = 'HCAT2.csv'
    dataname = 'AA_2022_EC21.shp'
    supportnames = [
        'AA_2022_EC21.cpg',
        'AA_2022_EC21.dbf',
        'AA_2022_EC21.prj',
        'AA_2022_EC21.shx',
    ]
    zipfilename = 'AA.zip'

    # create crop type data
    geojson_data = create_data_file(dataname)

    # archive the geojson to zip
    with zipfile.ZipFile(zipfilename, 'w') as zipf:
        zipf.write(dataname)
        for name in supportnames:
            zipf.write(name)

    # create csv metadata file
    create_csv(csvname)
