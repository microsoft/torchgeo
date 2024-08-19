#!/usr/bin/env python3

# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import csv
import hashlib
import os
import zipfile

import fiona
from rasterio.crs import CRS
from shapely.geometry import Polygon, mapping

# Size of example crop field polygon in projection units.
# This is set to align with Sentinel-2 test data, which is a 128x128 image at 10
# projection units per pixel (1280x1280 projection units).
SIZE = 1280


def create_data_file(dataname: str) -> None:
    schema = {'geometry': 'Polygon', 'properties': {'EC_hcat_c': 'str'}}
    with fiona.open(
        dataname, 'w', crs=CRS.from_epsg(32616), driver='ESRI Shapefile', schema=schema
    ) as shpfile:
        coordinates = [[0.0, 0.0], [0.0, SIZE], [SIZE, SIZE], [SIZE, 0.0], [0.0, 0.0]]
        # The offset aligns with tests/data/sentinel2/data.py.
        offset = [399960, 4500000 - SIZE]
        coordinates = [[x + offset[0], y + offset[1]] for x, y in coordinates]

        polygon = Polygon(coordinates)
        properties = {'EC_hcat_c': '1000000010'}
        shpfile.write({'geometry': mapping(polygon), 'properties': properties})


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
    os.remove(dataname)
    for name in supportnames:
        os.remove(name)

    # create csv metadata file
    create_csv(csvname)

    # Compute checksums
    with open(zipfilename, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{zipfilename}: {md5}')
    with open(csvname, 'rb') as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        print(f'{csvname}: {md5}')
